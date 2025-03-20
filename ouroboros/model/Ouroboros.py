import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from .modules import (
    optimizers_dict,
    MultiPropDecoder,
    activation_dict
)
from utils.metrics import reduce_dimension, cluster_models, reduce_2d
import rdkit.Chem.rdFMCS as FMCS
from .GeminiMol import GeminiMol
from .MolecularGenerator import MolecularGenerator
from rdkit import Chem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
import concurrent.futures
from utils.chem import gen_standardize_smiles, check_smiles_validity
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from tqdm import tqdm

class Ouroboros(GeminiMol):
    def __init__(
        self,
        model_name,
        batch_size = 512,
        predictor_info = {}, # key is proptery score, value is target score
        generator = True,
        mol4seed = True,
        driver = "AdamW",
        driver_params = {
            'replica_num': 1,
            'learning_rate': 2.0e-5,
            'num_steps_per_replica': 600,
            'loud': 0.2,
            'temperature': 0.40,
            'step_interval': 10
        },
        flooding = [0.3, 0.6], # CSS sim, 2D sim, min sim
        threads = 40
    ):
        # basic setting
        torch.set_float32_matmul_precision('high')
        super().__init__(
            model_path = model_name, 
            batch_size = batch_size,
            cache = True,
            threads = threads
        )
        self.Encoder.eval()
        self.pooling.eval()
        if os.path.exists(f'{self.model_name}/feat_stat.csv'):
            feat_stat = pd.read_csv(f"{self.model_name}/feat_stat.csv")
            self.encoding_profile = {
                'Mean': torch.tensor(feat_stat['Mean'].values, dtype=torch.float).unsqueeze(0).cuda(),
                'Min': torch.tensor(feat_stat['Min'].values, dtype=torch.float).unsqueeze(0).cuda(),
                'Max': torch.tensor(feat_stat['Max'].values, dtype=torch.float).unsqueeze(0).cuda(),
                'Std': torch.tensor(feat_stat['Std'].values, dtype=torch.float).unsqueeze(0).cuda(),
            }
        else:
            self.encoding_profile = None
        if generator:
            generator_params = json.load(open(f'{self.model_path}/generator_config.json', 'r'))
            self.Generator = MolecularGenerator(
                    encoding_size = self.encoder_params['encoding_features'], 
                    vocab_dict = json.load(open(f'{self.model_path}/vocabularies.json', 'r')),
                    chemical_language = generator_params['chemical_language'],
                    params = {
                        "num_heads": generator_params['num_heads'],
                        "num_layers": generator_params['num_layers'],
                        "activation": generator_params['activation']
                    }
                )
            self.Generator.load_state_dict(torch.load(f'{self.model_path}/Generator.pt'))
            self.Generator.cuda()
            self.Generator.eval()
        self.Predictor_Info = {}
        self.Predictor_Labels = []
        if len(list(predictor_info.keys())) > 0:
            self.projector = nn.ModuleDict()
            self.Predictor = nn.ModuleDict()
            for name in list(predictor_info.keys()):
                self.load_predictor(
                    predictor_name = name
                )
            self.projector.cuda()
            self.Predictor.cuda()
            self.projector.eval()
            self.Predictor.eval()
            self.goals = torch.tensor(
                list(predictor_info.values()), dtype=torch.float
            ).unsqueeze(0).cuda()
        self.mol4seed = mol4seed
        self.driver = driver
        self.driver_params = driver_params
        self.flooding = flooding

    def load_predictor(
        self,
        predictor_name = 'QSAR',
    ):
        self.Predictor_Info[predictor_name] = json.load(
            open(
                f'{self.model_name}/{predictor_name}/predictor.json', 
                'r'
        ))
        label_list = list(self.Predictor_Info[predictor_name]['label_dict'].keys())
        if len(label_list) == 1:
            label_list = [predictor_name]
        params = json.load(
            open(
                f'{self.model_name}/{predictor_name}/model_config.json', 
                'r'
        ))
        self.projector[predictor_name] = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(
                self.encoder_params['encoding_features'], 
                12288,
            ),
            activation_dict[params['activation']],
        )
        if os.path.exists(f'{self.model_name}/{predictor_name}/Projector.pt'):
            self.projector[predictor_name].load_state_dict(
                torch.load(f'{self.model_name}/{predictor_name}/Projector.pt')
        )
        self.Predictor[predictor_name] = MultiPropDecoder(
            feature_dim = (self.encoder_params['encoding_features'], 12288),
            output_dim = len(label_list),
            **params
        )
        self.Predictor[predictor_name].load_state_dict(
            torch.load(f'{self.model_name}/{predictor_name}/Decoder.pt')
        )
        self.Predictor[predictor_name].eval()
        self.Predictor_Labels += label_list

    def prepare(
        self,
        dataset,
        smiles_column = 'SMILES'
    ):
        dataset = dataset.dropna(subset=[smiles_column])
        print(f"NOTE: read the dataset size ({len(dataset)}).")
        dataset[smiles_column] = dataset[smiles_column].apply(
            lambda x: check_smiles_validity(x)
        )
        dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
        dataset.drop_duplicates(subset=[smiles_column], keep='first', inplace=True)
        print(f"NOTE: processed dataset size ({len(dataset)}).")
        dataset.reset_index(drop=True, inplace=True)
        return dataset

    def profiling_encoding(
        self,
        dataset,
        smiles_column = 'SMILES',
    ):
        dataset = self.prepare(
            dataset, smiles_column = smiles_column
        )
        dataset = self.extract(
            dataset,
            smiles_column
        )
        features_columns = dataset.columns
        print(dataset.head())
        print(f"NOTE: {self.model_name} encoding finished.")
        mean_values = dataset[features_columns].mean()
        std_values = dataset[features_columns].std()
        max_values = dataset[features_columns].max()
        min_values = dataset[features_columns].min()
        result_df = pd.DataFrame({'Mean': mean_values, 'Std': std_values, 'Max': max_values, 'Min': min_values})
        result_df['ExtremeVariance'] = result_df['Max'] - result_df['Min']
        result_df.to_csv(f'{self.model_name}/feat_stat.csv', index=True)
        result_df.sort_values(by=['ExtremeVariance'], ascending=False, inplace=True)
        result_df = result_df.reset_index()
        print(result_df)
        # plot shade range graph for Mean and lines for Max/Min
        plt.figure(figsize=(3.6*1.0, 3.2*1.0), dpi=600) # 
        plt.plot(result_df.index, result_df['Mean'], color='#588AF5', label='Mean', linewidth=0.3)
        plt.plot(result_df.index, result_df['Max'], color='#D861F5', label='Max', linewidth=0.3)
        plt.plot(result_df.index, result_df['Min'], color='#F55A46', label='Min', linewidth=0.3)
        plt.fill_between(result_df.index, result_df['Mean'] + result_df['Std'], result_df['Mean'] - result_df['Std'], color='#F5D690', linewidth=0, alpha=0.8)
        plt.fill_between(result_df.index, result_df['Mean'] + 3 * result_df['Std'], result_df['Mean'] - 3* result_df['Std'], color='#6CF5B2', linewidth=0, alpha=0.3)
        plt.ylim(result_df['Min'].min(), result_df['Max'].max())
        plt.xlim(0, len(result_df.index)+1)
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(f'{self.model_name}/feat_stat.png')
        plt.close()
        return result_df

    def reduce_dimension_plot(
        self, 
        dataset,
        smiles_column = 'SMILES',
        label_column = 'label', 
        output_fn = None, 
        method = 'tSNE', 
        concise = False,
        concise_max_feat = 100,
        plot_dim = [0, 1]
    ):
        label_series = dataset[label_column]
        features_dataset = self.extract(
            dataset,
            smiles_column
        )
        if concise and method == 'tSNE':
            std_values = features_dataset.std()
            std_top = std_values.sort_values(
                ascending = False
            )[:max(len(features_dataset.columns)//10, concise_max_feat)].index.to_list()
            X_embedded = features_dataset[std_top].values
        elif concise and method != 'tSNE':
            X_embedded = reduce_dimension[method](
                features_dataset.values, 
                min(len(dataset), len(features_dataset.columns)//10), 
                1207
            ) # random seed 1207
        else:
            X_embedded = features_dataset.values
        X_embedded = reduce_2d[method](
            X_embedded, 
            1207
        )  # random seed 1207
        if output_fn is None:
            output_fn = f"{self.model_name}_{method}"
        plt.figure(figsize=(4.5, 4.5), dpi=600)
        point_size = 6 + min(1200/len(label_series), 20)
        if all(isinstance(x, (float, int)) for x in label_series.to_list()):
            if len(label_series) >= 30:
                label_series = label_series.apply(lambda x: round(x, 1))
            label_set = sorted(list(set(label_series.to_list())))
            colors = plt.cm.rainbow(np.linspace(0.15, 0.85, len(label_set)))
        else:
            label_set = [label[0] for label in sorted(
                label_series.value_counts().to_dict().items(), key=lambda x: x[1], reverse=True
            )]
            colors = plt.cm.rainbow(np.linspace(0.15, 0.85, len(label_set)))
        color_id = 0
        for label in label_set:
            idx = (label_series.to_numpy() == label).nonzero()
            plt.scatter(X_embedded[idx, plot_dim[0]], X_embedded[idx, plot_dim[1]], c=colors[color_id], label=f'label={label}', marker = '.', s=point_size)
            color_id += 1
        if len(label_set) <= 10:
            plt.legend(loc='best')
        plt.title(f"{output_fn}")
        plt.tight_layout()
        plt.savefig(f"{output_fn}.png")
        plt.close()
        X_embedded = pd.DataFrame(X_embedded, columns=[f'{method}_1', f'{method}_2'])
        return X_embedded.join(label_series, how='left')

    def cluster(
        self, 
        dataset,
        smiles_column = 'SMILES',
        label_column = None,
        algorithm = 'AffinityPropagation',
        num_clusters = 2
    ):
        features_dataset = self.extract(
            dataset,
            smiles_column
        )
        X_embedded = reduce_dimension['PCA'](
            features_dataset.values, 
            len(features_dataset.columns)//10, 
            1207
        ) # random seed 1207
        cluster_model = cluster_models[algorithm](num_clusters).fit(X_embedded)
        pred_labels = cluster_model.labels_
        dataset = dataset.join(
            pd.DataFrame(
                pred_labels, 
                columns=[f'{algorithm}_ID']
            ), 
            how='left'
        )
        if label_column is not None:
            labels = dataset[label_column].tolist()
            cm = confusion_matrix(list(map(str, labels)), list(map(str, pred_labels)))
            purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
            rand_score = adjusted_rand_score(labels, pred_labels)
            ari = adjusted_rand_score(labels, pred_labels)
            print(f'NOTE: {algorithm} cluster purity: {purity}, rand_score: {rand_score}, adjusted_rand_score: {ari}')
        return dataset

    def predict_similarity(
        self, 
        smiles1_list, 
        smiles2_list,
        as_pandas = True, 
        similarity_metrics = ['Cosine']
    ):
        self.eval()
        assert len(smiles1_list) == len(smiles2_list), f'Error: smiles list must be same length.'
        with torch.no_grad():
            pred_values = {key:[] for key in similarity_metrics}
            for i in range(0, len(smiles1_list), self.batch_size):
                sent1 = smiles1_list[i:i+self.batch_size]
                sent2 = smiles2_list[i:i+self.batch_size]
                encoding_1 = self.encode(sent1)
                encoding_2 = self.encode(sent2)
                # Concatenate input sentences
                for label_name in similarity_metrics:
                    pred = self.molecular_comparison(
                        encoding_1, 
                        encoding_2, 
                        label_name
                    )
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values)
                return res_df
            else:
                return pred_values

    def predict_scores(
        self, 
        df, 
        smiles_column = 'smiles',
    ):
        with torch.no_grad():
            res_list = []
            # Encode all sentences using the encoder
            for predictor_name in self.Predictor_Info.keys():
                label_list = list(self.Predictor_Info[predictor_name]['label_dict'].keys())
                if len(label_list) == 1:
                    label_list = [predictor_name]
                pred_values = {key:[] for key in label_list}
                for i in tqdm(range(0, len(df), self.batch_size), desc=f'Predicting {predictor_name}'):
                    rows = df.iloc[i:i+self.batch_size]
                    total_smiles = rows[smiles_column].to_list()
                    features = self.encode(total_smiles)
                    projected_feats = self.projector[predictor_name](features)
                    # Calculate scores of encoded smiles
                    pred = self.Predictor[predictor_name](
                        features, projected_feats
                    ).cpu().detach().numpy()
                    for k in range(len(label_list)):
                        pred_values[label_list[k]] += list(pred[:, k])
                res_list.append(pd.DataFrame(pred_values, columns=label_list))
        return df.join(pd.concat(res_list, axis=1), how='left')

    def similarity_matrix(
        self, 
        dataset,
        smiles_column = 'smiles',
        label_column = 'label',
        id_column = 'ID',
        reference_label = 'A',
        query_label = 'B',
        output_fn = None
    ):
        dataset = self.prepare(
            dataset, smiles_column = smiles_column
        )
        if output_fn is None:
            output_fn = self.model_name
        ref_smiles = dataset[dataset[label_column]==reference_label][smiles_column].tolist()
        query_smiles = dataset[dataset[label_column]==query_label][smiles_column].tolist()
        ref_feats = self.encode(ref_smiles).unsqueeze(1).expand(-1, len(query_smiles), -1)
        query_feats = self.encode(query_smiles).unsqueeze(0).expand(len(ref_smiles), -1, -1)
        matrix = torch.nn.functional.cosine_similarity(ref_feats, query_feats, dim=-1)
        plt.figure(figsize=(6.9, 6.2), dpi=600)
        score_array = matrix.cpu().detach().numpy()
        plt.imshow(score_array, vmin=0.0, vmax=1.0)
        if id_column in dataset.columns:
            ref_ids = dataset[dataset[label_column]==reference_label][id_column].tolist()
            query_ids = dataset[dataset[label_column]==query_label][id_column].tolist()
        else:
            ref_ids = ref_smiles
            query_ids = query_smiles
        if len(query_ids) <= 20:
            plt.xticks(range(len(query_ids)), query_ids, rotation=45)
        if len(ref_ids) <= 20:
            plt.yticks(range(len(ref_ids)), ref_ids)
        plt.title(f"{output_fn}_Matrix", fontsize='large', fontweight='bold')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{output_fn}_Matrix.png")
        plt.close()
        return pd.DataFrame(
            matrix.cpu().detach().numpy(),
            index = ref_ids,
            columns = query_ids
        )

    def create_database(
        self, 
        features_database, 
        smiles_column='smiles', 
        worker_num=1
    ):
        data = features_database[smiles_column].tolist()
        features_database['features'] = None
        self.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(data), 2*worker_num*self.batch_size), desc = 'Encoding'):
                smiles_list = data[i:i+2*worker_num*self.batch_size]
                features = self.encode(smiles_list)
                features_list = list(features.cpu().detach().numpy())
                for j in range(len(features_list)):
                    features_database.at[i+j, 'features'] = features_list[j]
        return features_database

    def molecular_comparison(
        self, 
        features,
        ref_features,
        mode = 'Cosine'
    ):
        if mode == 'Cosine':
            return torch.nn.functional.cosine_similarity(features, ref_features, dim=-1)
        elif mode == 'Pearson':
            encoded1_normalized = (
                features - torch.mean(features, dim=-1, keepdim=True)
                ) / (
                    torch.std(features, dim=-1, keepdim=True) + 1e-7
                )
            encoded2_normalized = (
                ref_features - torch.mean(ref_features, dim=-1, keepdim=True)
                ) / (
                    torch.std(ref_features, dim=-1, keepdim=True) + 1e-7
                )
            return torch.mean(encoded1_normalized * encoded2_normalized, dim=-1)
        elif mode == 'Euclidean':
            return 1 - torch.sqrt(torch.sum((features - ref_features)**2, dim=-1))
        elif mode == 'Manhattan':
            return 1 - torch.sum(torch.abs(features - ref_features), dim=-1)
        else:
            raise ValueError("Error: unkown mode: %s" % mode)

    def database_screening(
        self, 
        features_database, 
        ref_smiles, 
        as_pandas = True, 
        similarity_metrics=['Cosine'], 
        worker_num=1
    ):
        torch.set_float32_matmul_precision('high')
        features_list = features_database['features'].tolist()
        with torch.no_grad():
            pred_values = {key:[] for key in similarity_metrics}
            ref_features = self.encode([ref_smiles]).cuda()
            for i in range(0, len(features_list), worker_num*self.batch_size):
                features_batch = features_list[i:i+worker_num*self.batch_size]
                query_features = torch.from_numpy(np.array(features_batch)).cuda()
                for label_name in similarity_metrics:
                    pred = self.molecular_comparison(
                        query_features, 
                        ref_features.repeat(len(features_batch), 1), 
                        label_name
                    )
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=similarity_metrics)
                return res_df
            else:
                return pred_values 

    def virtual_screening(
        self, 
        ref_smiles_list, 
        query_smiles_table, 
        input_with_features = False,
        properties_prediction = True,
        smiles_column = 'smiles', 
        return_all_col = True,
        similarity_metrics = None, 
        worker_num = 1
    ):
        if input_with_features:
            features_database = query_smiles_table
        else:
            features_database = self.create_database(
                query_smiles_table, 
                smiles_column = smiles_column, 
                worker_num = worker_num
            )
        properties_names = []
        if properties_prediction and len(self.Predictor_Info) > 0:
            with torch.no_grad():
                properties_list = []
                features_list = features_database['features'].tolist()
                for predictor_name in self.Predictor_Info.keys():
                    label_list = list(self.Predictor_Info[predictor_name]['label_dict'].keys())
                    if len(label_list) == 1:
                        label_list = [predictor_name]
                    pred_values = {key:[] for key in label_list}
                    for i in range(0, len(features_list), worker_num*self.batch_size):
                        features_batch = features_list[i:i+worker_num*self.batch_size]
                        features = torch.from_numpy(np.array(features_batch)).cuda()
                        projected_feats = self.projector[predictor_name](features)
                        pred = self.Predictor[predictor_name](
                            features, projected_feats
                        ).cpu().detach().numpy()
                        for k in range(len(label_list)):
                            pred_values[label_list[k]] += list(pred[:, k])
                    properties_list.append(pd.DataFrame(pred_values, columns=label_list))
                    properties_names += label_list
                features_database = features_database.join(pd.concat(properties_list, axis=1), how='left')
            if len(ref_smiles_list) == 0:
                total_res = features_database.copy()
                return total_res
        total_res = pd.DataFrame()
        for ref_smiles in ref_smiles_list:
            query_scores = self.database_screening(
                features_database, 
                ref_smiles, 
                as_pandas=True, 
                similarity_metrics=similarity_metrics
            )
            assert len(query_scores) == len(features_database), f"Error: different length between original dataframe with predicted scores! {ref_smiles}"
            if return_all_col:
                total_res = pd.concat([
                    total_res, 
                    features_database.copy().join(query_scores, how='left')
                ], ignore_index=True)
            else:
                total_res = pd.concat([
                    total_res, 
                    features_database[[smiles_column]+properties_names].copy().join(query_scores, how='left')
                ], ignore_index=True)
        return total_res

    def clean_smiles_list(self, input_sents):
        output_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
            futures = []
            for smiles in input_sents:
                    futures.append(executor.submit(gen_standardize_smiles, smiles))
            for future in concurrent.futures.as_completed(futures):
                output_list.append(future.result())
        return [item for item in output_list if item != 'smiles_unvaild'] 

    def generate(
        self, 
        sampling_tensor, 
        smiles_len = 256,
        temperature = 0.10
    ):
        return self.Generator.decoder(
            sampling_tensor, 
            smiles_len=smiles_len, 
            temperature=temperature
        )

    def assess_decode_smiles_(
        self, 
        decoded_smiles, 
        input_smiles, 
        mode = 'AtomPair', # MCS, AtomPair
    ):
        try:
            mol = Chem.MolFromSmiles(decoded_smiles)
            ref_mol = Chem.MolFromSmiles(input_smiles)
            if mode == 'AtomPair':
                similarity = DataStructs.FingerprintSimilarity(
                    GetHashedAtomPairFingerprintAsBitVect(mol), 
                    GetHashedAtomPairFingerprintAsBitVect(ref_mol), 
                )
            elif mode == 'MCS':
                similarity = FMCS.FindMCS(
                    [mol, ref_mol], 
                    ringMatchesRingOnly = False, 
                    atomCompare=(
                        FMCS.AtomCompare.CompareElements
                    )
                ).numBonds / ref_mol.GetNumBonds()
            return similarity
        except:
            return 0.0

    def assess_decode_smiles(
            self, 
            decoded_smiles, 
            input_smiles, 
            threads = 24
        ):
        # calculate the validity and similarity
        similarities = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for input, decoded in zip(input_smiles, decoded_smiles):
                futures.append(executor.submit(self.assess_decode_smiles_, input, decoded))
            for future in concurrent.futures.as_completed(futures):
                similarities.append(future.result())
        return similarities
        
    def evaluate_generator(
            self, 
            smiles_list, 
    ):
        with torch.no_grad():
            generated_smiles = []
            for start in range(0, len(smiles_list), self.batch_size):
                end = min(start + self.batch_size, len(smiles_list))
                current_input = smiles_list[start:end]
                total_features = self.encode(current_input)
                ## Mol Generator
                max_len = self.Generator.get_max_len(
                    current_input, 
                )
                generated_smiles += self.Generator.decoder(
                    total_features, 
                    smiles_len = max_len,
                    temperature = 0.00
                )
            gene_eva = pd.DataFrame(
                {
                    'generated_smiles': generated_smiles,
                    'original_smiles': smiles_list,
                    'similarity': self.assess_decode_smiles(generated_smiles, smiles_list)
                }
            )
            gene_eva.sort_values(
                'similarity',
                ascending = False,
                inplace = True
            )
        return gene_eva

    def add_nosie(
        self,
        sampling_tensor,
        loud = 0.1,
        mask_ratio = 0.5
    ):
        random_tensor = ( torch.randn_like(sampling_tensor).cuda() - 0.5 ) * (
            self.encoding_profile['Std'] * loud
        )
        mask = torch.rand_like(sampling_tensor).cuda() < mask_ratio
        random_tensor[mask] = 0
        return random_tensor + sampling_tensor

    def decode_molecules(
        self,
        sampling_tensor,
        target_lenght_list = [96, 128],
        temperature = 0.10
    ):
        with torch.no_grad():
            output_list = []
            for smiles_len in target_lenght_list:
                for i in self.generate(
                    sampling_tensor, 
                    smiles_len=smiles_len, 
                    temperature=temperature
                ):
                    try:
                        mol = Chem.MolFromSmiles(i)
                        Chem.SanitizeMol(mol)
                        smiles = Chem.MolToSmiles(
                            mol, 
                            kekuleSmiles=False, 
                            doRandom=False, 
                            isomericSmiles=True
                        )
                        if len(smiles) > 1:
                            output_list.append(smiles)
                        else:
                            continue
                    except:
                        continue
        return list(set(output_list))

    def ouroboros_check(
        self, 
        ref_smiles
    ):
        ref_smiles = gen_standardize_smiles(ref_smiles)
        sampling_tensor = self.encode([ref_smiles]).clone().detach()
        output_smiles = {
            'smiles': [],
            'temperature': [],
        }
        replica_tensor = sampling_tensor.clone().detach()
        for temperature in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            step_smiles_list = self.decode_molecules(
                replica_tensor.repeat(self.batch_size, 1),
                target_lenght_list = [ 
                    self.Generator.get_max_len([ref_smiles]) + 24
                ],
                temperature = temperature
            )
            output_smiles['smiles'] += step_smiles_list
            output_smiles['temperature'] += [temperature] * len(step_smiles_list)
            print(f'NOTE: total {len(set(step_smiles_list))} unique smiles under temperature {temperature}.')
        results = pd.DataFrame(output_smiles)
        results = self.virtual_screening(
            [ref_smiles], 
            results, 
            properties_prediction = False,
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = ['Cosine'], 
            worker_num = 1
        )
        print(f'NOTE: check the accessibility of {ref_smiles}, total {len(set(output_smiles["smiles"]))} unique smiles.')
        print(f'NOTE: the miximum simlarity: {results["Cosine"].max()}, mean simlarity: {results["Cosine"].mean()}')
        del results['features']
        return results

    def stochastic_propagation(
        self, 
        start_smiles,
        temperature = 0.4,
        replica_num = 1,
        num_steps_per_replica = 200,
        loud = 0.05
    ):
        sampling_tensor = self.encode([start_smiles]).clone().detach() if self.mol4seed else self.encoding_profile['Mean'].clone().detach()
        length_list = [ 
            self.Generator.get_max_len([start_smiles]) + 16
        ]
        output_smiles = {
            'smiles': [],
            'replica_id': [],
            'step': []
        }
        step_smiles_list = self.decode_molecules(
            sampling_tensor.clone().detach().repeat(self.batch_size, 1),
            target_lenght_list = length_list,
            temperature = 0.0
        )
        output_smiles['smiles'] += step_smiles_list
        output_smiles['replica_id'] += [0] * len(step_smiles_list)
        output_smiles['step'] += [0] * len(step_smiles_list)
        for replica in range(replica_num):
            replica_tensor = sampling_tensor.clone().detach()
            for step in range(num_steps_per_replica):
                replica_tensor = self.add_nosie(replica_tensor, loud = loud)
                step_smiles_list = self.decode_molecules(
                    replica_tensor.repeat(self.batch_size, 1),
                    target_lenght_list = length_list,
                    temperature = temperature
                )
                output_smiles['smiles'] += step_smiles_list
                output_smiles['replica_id'] += [replica+1] * len(step_smiles_list)
                output_smiles['step'] += [step+1] * len(step_smiles_list)
                print(f'replica {replica+1}, step {step+1}, new {len(step_smiles_list)}, total {len(set(output_smiles["smiles"]))}.')
        output = pd.DataFrame(output_smiles)
        output.drop_duplicates(
            subset = ['smiles'], 
            keep = 'first', 
            inplace = True,
            ignore_index = True
        )
        results = self.virtual_screening(
            [start_smiles], 
            output, 
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = ['Cosine'], 
            worker_num = 1
        )
        del results['features']
        return results

    def MCMC(
        self, 
        sampling_tensor, 
        task_function, 
        length_list = [128, 256],
        replica_num=10, 
        num_steps_per_replica=30, 
        loud = 0.5,
        temperature = 0.10,
        learning_rate = None,
    ):
        output_smiles = {
            'smiles': [],
            'replica_id': [],
            'step': [],
        }
        for replica in range(replica_num):
            replica_tensor = sampling_tensor.clone().detach()
            for step in range(num_steps_per_replica):
                replica_tensor = self.add_nosie(replica_tensor, loud = loud*(1-step/(num_steps_per_replica+1)))
                loss = task_function(
                    replica_tensor, 
                )
                step_smiles_list = self.decode_molecules(
                    replica_tensor.repeat(self.batch_size, 1),
                    target_lenght_list = length_list,
                    temperature = temperature
                )
                print(f'replica {replica+1}, step {step+1}, loss {torch.mean(loss)}, new {len(step_smiles_list)}, total {len(set(output_smiles["smiles"]))}.')
                output_smiles['smiles'] += step_smiles_list
                output_smiles['replica_id'] += [replica+1] * len(step_smiles_list)
                output_smiles['step'] += [step+1] * len(step_smiles_list)
        return pd.DataFrame(output_smiles)

    def directed_evolution(
        self, 
        sampling_tensor, 
        task_function, 
        length_list = [128, 256], 
        optim_type = 'AdamW', 
        replica_num = 5, 
        num_steps_per_replica = 300, 
        loud = 0.5,
        temperature = 0.10,
        max_patience = 100,
        learning_rate = 2.0e-5,
        step_interval = 10
    ):
        # Load the models and optimizer
        if optim_type not in optimizers_dict:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        output_smiles = {
            'smiles': [],
            'replica_id': [],
            'step': [],
        }
        for replica in range(replica_num):
            if replica == 0:
                replica_tensor = sampling_tensor.clone().detach().requires_grad_(True)
            elif replica > 0:
                replica_tensor = self.add_nosie(
                    sampling_tensor.clone().detach(),
                    loud = loud,
                    mask_ratio = 0.5
                ).requires_grad_(True)
            optimizer = optimizers_dict[optim_type]([replica_tensor], lr=learning_rate)
            best_score = 999
            patience = max_patience
            for step in range(num_steps_per_replica):
                replica_encoding = replica_tensor.clone().detach()
                optimizer.zero_grad()
                loss = task_function(
                    replica_tensor
                ).mean()
                loss.backward(retain_graph=True)
                optimizer.step()
                score = loss.item()
                if score < best_score:
                    best_score = score
                else:
                    patience -= 1
                if (step+1) % step_interval == 0:
                    decode_smiles_list = self.decode_molecules(
                        replica_encoding.repeat(self.batch_size, 1),
                        target_lenght_list = length_list,
                        temperature = temperature
                    )
                    output_smiles['smiles'] += decode_smiles_list
                    output_smiles['replica_id'] += [replica+1] * len(decode_smiles_list)
                    output_smiles['step'] += [step+1] * len(decode_smiles_list)
                    print(f'replica {replica+1}, step {step+1}, loss {score}, new {len(decode_smiles_list)}, total {len(set(output_smiles["smiles"]))}')
                    if len(decode_smiles_list) == 0:
                        patience -= 1
                    if patience <= 0:
                        break
        return pd.DataFrame(output_smiles)

    def proptery_scorer(
        self,
        features,
    ):
        pred_values_list = []
        for predictor_name in self.Predictor_Info.keys():
            # Calculate scores of encoded smiles
            projected_feats = self.projector[predictor_name](features)
            # Calculate scores of encoded smiles
            pred_matrix = self.Predictor[predictor_name](features, projected_feats)
            pred_values_list.append(pred_matrix)
        total_matrix = torch.cat(
            pred_values_list,
            dim = -1
        )
        return total_matrix

    def molecular_generation(
        self,
        start_features,
        func,
        length_list = [128, 256],
    ):
        if self.driver in ['AdamW', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adamax', 'Rprop', 'NAdam']:
            output = self.directed_evolution(
                start_features, 
                func, 
                length_list = length_list,
                optim_type = self.driver, 
                **self.driver_params
            )
        elif self.driver == 'MCMC':
            output = self.MCMC(
                start_features,
                func, 
                length_list = length_list,
                **self.driver_params
            )
        else:
            raise NotImplementedError(f"Driver {self.driver} is not implemented.")
        output.drop_duplicates(
            subset = ['smiles'], 
            keep = 'first', 
            inplace = True,
            ignore_index = True
        )
        return output

    def directional_optimization(
        self,
        features,
    ):
        pred_labels = self.proptery_scorer(features)
        abs_property_loss = (pred_labels - self.goals).abs()
        property_loss = torch.mean(
            (
                -1 / ( 1 + 10000 ** (abs_property_loss - 0.5)) + 1 
                # 0.5 is the Maximum Slope Point, all properties are normalized to [0, 1] in property prediction
            ), 
            dim = -1
        )
        return property_loss

    def scaffold_hopping(
        self,
        features,
        ref_features,
    ):
        return (
            self.pairwise_similarity(features, ref_features) - self.flooding[1]
        ).abs()

    def directional_scaffold_hopping(
        self,
        features,
        ref_features,
    ):
        similarity_loss = self.scaffold_hopping(features, ref_features)
        return similarity_loss * self.directional_optimization(features)

    def directed_exploration(
        self,
        start_smiles = None,
        exploration_obj = 'migration',
    ):
        if self.mol4seed and start_smiles is not None:
            assert isinstance(start_smiles, str), f'Please provide a SMILES for mol4seed.'
            start_features = self.encode(
                [start_smiles]
            ).clone().detach()
            length_list = [
                self.Generator.get_max_len([start_smiles]) + 24
            ]
            ref_smiles_list = [start_smiles] # only properties prediction
        else:
            start_features = self.encoding_profile['Mean'].clone().detach()
            length_list = [ 80 ]
            ref_smiles_list = [] 
        if exploration_obj == 'directional_optimization':
            output = self.molecular_generation(
                start_features,
                partial(
                    self.directional_optimization, 
                ),
                length_list = length_list
            )
        elif exploration_obj == 'scaffold_hopping':
            output = self.molecular_generation(
                start_features,
                partial(
                    self.scaffold_hopping, 
                    ref_features = start_features.clone().detach()
                ),
                length_list = length_list
            )
        elif exploration_obj == 'directional_scaffold_hopping':
            output = self.molecular_generation(
                start_features,
                partial(
                    self.directional_scaffold_hopping, 
                    ref_features = start_features.clone().detach()
                ),
                length_list = length_list
            )
        else:
            raise NotImplementedError(f"Migration object {exploration_obj} is not implemented.")
        if len(self.Predictor_Info) != 0:
            results = self.virtual_screening(
                ref_smiles_list,
                output, 
                smiles_column = 'smiles', 
                return_all_col = True,
                similarity_metrics = ['Cosine'], 
                worker_num = 1
            )
        del results['features']
        return results

    def migration(
        self,
        features,
        ref_features,
    ):
        return (1 - self.pairwise_similarity(ref_features, features))

    def restricted_migration(
        self,
        features,
        ref_features,
    ):
        similarity_loss = (1 - self.pairwise_similarity(ref_features, features))
        pred_labels = self.proptery_scorer(features)
        abs_property_loss = (pred_labels - self.goals).abs()
        property_loss = torch.mean(
            (
                -1 / ( 1 + 10000 ** (abs_property_loss - 0.5)) + 1 
                # 0.5 is the threshold, all properties are normalized to [0, 1] in property prediction
            ), 
            dim = -1
        )
        return similarity_loss * property_loss

    def directed_migration(
        self,
        ref_smiles,
        start_smiles = None,
        migration_obj = 'migration',
    ):
        ref_features = self.encode([ref_smiles]) 
        if start_smiles is None:
            start_features = self.encoding_profile['Mean']
            length_list = [ 
                self.Generator.get_max_len([ref_smiles]) + 24
            ]
        else:
            start_features = self.encode(
                [start_smiles]
            ).clone().detach() if self.mol4seed else self.encoding_profile['Mean'].clone().detach()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles, start_smiles]) + 24
            ]
        if migration_obj == 'migration':
            output = self.molecular_generation(
                start_features,
                partial(
                    self.migration, 
                    ref_features=ref_features
                ),
                length_list = length_list
            )
        elif migration_obj == 'restricted_migration':
            output = self.molecular_generation(
                start_features,
                partial(
                    self.restricted_migration, 
                    ref_features=ref_features,
                ),
                length_list = length_list
            )
        elif migration_obj == 'scaffold_hopping':
            output = self.molecular_generation(
                ref_features,
                partial(
                    self.scaffold_hopping, 
                    ref_features=ref_features
                ),
                length_list = length_list
            )
        else:
            raise NotImplementedError(f"Migration object {migration_obj} is not implemented.")
        results = self.virtual_screening(
            [ref_smiles], 
            output, 
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = ['Cosine'], 
            worker_num = 1
        )
        del results['features']
        return results
    
    def fusion(
        self,
        features,
        ref_features,
        temperature = 0.1
    ):
        loss = torch.tensor(0.0).cuda()
        for ref_feats in ref_features:
            expand_features = features.repeat(ref_feats.size(0), 1)
            fusion_similarity = self.pairwise_similarity(expand_features, ref_feats)
            sigmoid_loss = ( 1 / (
                1 + 10000 ** (fusion_similarity - self.flooding[0]))
            )
            if temperature == 0.0:
                loss += sigmoid_loss.min()
            else:
                softmin_loss = 1-torch.logsumexp(-sigmoid_loss / temperature, dim=0)
                loss += softmin_loss
        return loss

    def directional_fusion(
        self,
        features,
        ref_features,
        temperature = 0.1
    ):
        similarity_loss = self.fusion(features, ref_features, temperature)
        pred_labels = self.proptery_scorer(features)
        abs_property_loss = (pred_labels - self.goals).abs()
        property_loss = torch.mean(
            (
                -1 / ( 1 + 10000 ** (abs_property_loss - 0.5)) + 1 
                # 0.5 is the threshold, all properties are normalized to [0, 1] in property prediction
            ), 
            dim = -1
        )
        return similarity_loss + property_loss

    def chemical_fusion(
        self,
        ref_smiles = {},
        temperature = 0.1
    ):
        ref_smiles_list = []
        for smiles_list in ref_smiles.values():
            ref_smiles_list += smiles_list
        length_list = [ 
            self.Generator.get_max_len(ref_smiles_list) + 24
        ]
        if self.mol4seed == False:
            ref_features = []
            for smiles_list in ref_smiles.values():
                ref_features.append(self.encode(smiles_list))
            start_features = self.encoding_profile['Mean'].clone().detach()
            output = self.molecular_generation(
                start_features,
                partial(
                    self.fusion, 
                    ref_features = ref_features,
                    temperature = temperature
                ) if len(self.Predictor_Info) == 0 else partial(
                    self.directional_fusion,
                    ref_features = ref_features,
                    temperature = temperature
                ),
                length_list = length_list
            )
        else:
            total_output = []
            for start_smiles in ref_smiles_list:
                start_features = self.encode(
                    [start_smiles]
                ).clone().detach()
                ref_features = []
                for smiles_list in ref_smiles.values():
                    ref_features.append(self.encode(list(set(smiles_list) - set([start_smiles]))))
                output = self.molecular_generation(
                    start_features,
                    partial(
                        self.fusion, 
                        ref_features = ref_features,
                        temperature = temperature
                    ) if len(self.Predictor_Info) == 0 else partial(
                        self.directional_fusion,
                        ref_features = ref_features,
                        temperature = temperature
                    ),
                    length_list = length_list
                )
                output['start_smiles'] = start_smiles
                total_output.append(output)
                output = pd.concat(total_output, ignore_index = True)
        database = self.create_database(
            output, 
            smiles_column = 'smiles', 
            worker_num = 4
        )
        for target, smiles_list in ref_smiles.items():
            scores = self.virtual_screening(
                smiles_list, 
                database, 
                input_with_features = True,
                properties_prediction = False,
                smiles_column = 'smiles', 
                return_all_col = False,
                similarity_metrics = ['Cosine'], 
                worker_num = 1
            )
            scores = scores.groupby('smiles').agg({'Cosine': ['mean', 'max']})
            scores.columns = ['_'.join(map(str, col)).strip('_') for col in scores.columns]
            scores.reset_index(inplace = True)
            scores[f'{target}_mean'] = scores['Cosine_mean']
            scores[f'{target}_max'] = scores['Cosine_max']
            scores[target] = (scores[f'{target}_mean'] + scores[f'{target}_max'])/2
            output = pd.merge(
                output,
                scores[['smiles', target, f'{target}_max', f'{target}_mean']], 
                on = 'smiles'
            )
        output['Score'] = output[list(ref_smiles.keys())].apply(
            lambda row: (row - self.flooding[0]).clip(lower=0).sum(), axis=1
        )
        del output['features']
        return output
    
    
