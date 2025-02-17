import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import torch
import torch.nn as nn
import pandas as pd
from functools import partial
from .modules import (
    optimizers_dict,
    MultiPropDecoder
)
from .GeminiMol import GeminiMol
from .MolecularGenerator import MolecularGenerator
from rdkit import Chem
from tqdm import tqdm

class Ouroboros(GeminiMol):
    def __init__(
        self,
        model_name,
        batch_size = 512,
        predictor_info = {}, # key is proptery score, val is score weight (higher is better)
    ):
        # basic setting
        torch.set_float32_matmul_precision('high')
        super().__init__(
            model_path = model_name, 
            batch_size = batch_size,
            cache = True
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
        if len(list(predictor_info.keys())) > 0:
            self.projector = nn.ModuleDict()
            self.Predictor = nn.ModuleDict()
            self.Predictor_Info = {}
            self.Predictor_Labels = []
            for name in list(predictor_info.keys()):
                self.load_predictor(
                    predictor_name = name
                )
            self.projector.cuda()
            self.Predictor.cuda()
            self.projector.eval()
            self.Predictor.eval()
            self.label_weights = torch.tensor(
                list(predictor_info.values()), 
                dtype=torch.float
            ).unsqueeze(0).cuda()

    def load_predictor(
        self,
        predictor_name = 'QSAR',
    ):
        self.projector[predictor_name] = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.Linear(
                self.encoder_params['encoding_features'], 
                8192,
            ),
            nn.Sigmoid(),
        )
        if os.path.exists(f'{self.model_name}/{predictor_name}/Projector.pt'):
            self.projector[predictor_name].load_state_dict(
                torch.load(f'{self.model_name}/{predictor_name}/Projector.pt')
        )
        self.Predictor_Info[predictor_name] = json.load(
            open(
                f'{self.model_name}/{predictor_name}/predictor.json', 
                'r'
        ))
        label_list = list(self.Predictor_Info[predictor_name]['label_dict'].keys())
        if len(label_list) == 1:
            label_list = [predictor_name]
        self.Predictor[predictor_name] = MultiPropDecoder(
            feature_dim = self.encoder_params['encoding_features'] + 8192,
            output_dim = len(label_list),
            **json.load(
                open(
                    f'{self.model_name}/{predictor_name}/model_config.json', 
                    'r'
            ))
        )
        self.Predictor[predictor_name].load_state_dict(
            torch.load(f'{self.model_name}/{predictor_name}/Decoder.pt')
        )
        self.Predictor_Labels += label_list

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
                        torch.cat(
                            (projected_feats, features), dim = -1
                        )
                    ).cpu().detach().numpy()
                    for k in range(len(label_list)):
                        pred_values[label_list[k]] += list(pred[:, k])
                res_list.append(pd.DataFrame(pred_values, columns=label_list))
        return df.join(pd.concat(res_list, axis=1), how='left')

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

    def add_nosie(
        self,
        sampling_tensor,
        loud = 0.1,
        mask_ratio = 0.5
    ):
        random_tensor = ( torch.randn_like(sampling_tensor).cuda() - 0.5 ) * (self.encoding_profile['Std'] * loud)
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
        sampling_tensor = self.encode([ref_smiles]).clone().detach()
        output_smiles = {
            'smiles': [],
            'temperature': [],
            'noise': [],
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
            output_smiles['noise'] += [0.0] * len(step_smiles_list)
            print(f'NOTE: total {len(set(output_smiles["smiles"]))} unique smiles under temperature {temperature}.')
        for noise in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            replica_tensor = self.add_nosie(sampling_tensor, loud = noise).clone().detach()
            step_smiles_list = self.decode_molecules(
                replica_tensor.repeat(self.batch_size, 1),
                target_lenght_list = [ 
                    self.Generator.get_max_len([ref_smiles]) + 24
                ],
                temperature = 0.2
            )
            output_smiles['smiles'] += step_smiles_list
            output_smiles['temperature'] += [0.2] * len(step_smiles_list)
            output_smiles['noise'] += [noise] * len(step_smiles_list)
            print(f'NOTE: total {len(set(output_smiles["smiles"]))} unique smiles under noise {noise}.')
        results = pd.DataFrame(output_smiles)
        results = self.virtual_screening(
            [ref_smiles], 
            results, 
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = ['Pearson'], 
            worker_num = 1
        )
        return results

    def random_walking(
        self, 
        start_smiles,
        temperature = 0.4,
        replica_num = 1,
        num_steps_per_replica = 200,
        loud = 0.05
    ):
        sampling_tensor = self.encode([start_smiles]).clone().detach()
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
            temperature = 0.4
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
            similarity_metrics = ['Pearson'], 
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
            optim_type='AdamW', 
            replica_num = 5, 
            num_steps_per_replica = 300, 
            loud = 0.5,
            temperature = 0.10,
            max_patience = 12,
            learning_rate = 2.0e-5
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
            replica_tensor = self.add_nosie(
                sampling_tensor.clone().detach(),
                loud = loud,
                mask_ratio = 0.5
            ).requires_grad_(True) if replica > 0 else sampling_tensor.clone().detach().requires_grad_(True)
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
                    patience -= 3
                if patience <= 0:
                    break
        return pd.DataFrame(output_smiles)

    def molecular_generation(
        self,
        start_features,
        func,
        mode = 'directed_evolution',
        length_list = [128, 256]
    ):
        if mode in ['AdamW', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adamax', 'Rprop', 'NAdam']:
            output = self.directed_evolution(
                start_features, 
                func, 
                length_list = length_list,
                optim_type = mode, 
                replica_num = 3, 
                learning_rate = 2.0e-5,
                num_steps_per_replica = 600, 
                loud = 0.2,
                temperature = 0.40
            )
        elif mode == 'MCMC':
            output = self.MCMC(
                start_features,
                func, 
                length_list = length_list,
                replica_num = 20, 
                num_steps_per_replica = 50, 
                loud = 0.5,
                temperature = 0.60, 
            )
        return output

    def migration(
        self,
        features,
        ref_features,
    ):
        return 1 - self.pairwise_similarity(ref_features, features)

    def forward_migration(
        self,
        ref_smiles,
        start_smiles = None,
        mode = 'directed_evolution',
    ):
        ref_features = self.encode([ref_smiles]) 
        if start_smiles is None:
            start_features = self.encoding_profile['Mean'] * torch.randn_like(ref_features).cuda()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles]) + 24
            ]
        else:
            start_features = self.encode([start_smiles]).clone().detach()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles, start_smiles]) + 24
            ]
        output = self.molecular_generation(
            start_features,
            partial(self.migration, ref_features=ref_features),
            mode = mode,
            length_list = length_list
        )
        output.drop_duplicates(
            subset = ['smiles'], 
            keep = 'first', 
            inplace = True,
            ignore_index = True
        )
        results = self.virtual_screening(
            [ref_smiles], 
            output, 
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = ['Pearson'], 
            worker_num = 1
        )
        del results['features']
        return results
    
    def proptery_scorer(
        self,
        features,
    ):
        pred_values_list = []
        for predictor_name in self.Predictor_Info.keys():
            # Calculate scores of encoded smiles
            projected_feats = self.projector[predictor_name](features)
            # Calculate scores of encoded smiles
            pred_matrix = self.Predictor[predictor_name](
                torch.cat(
                    (projected_feats, features), dim = -1
                )
            )
            pred_values_list.append(pred_matrix)
        total_matrix = torch.cat(
            pred_values_list,
            dim = -1
        )
        return total_matrix * self.label_weights

    def scaffold_hopping(
            self,
            features,
            ref_features,
            flooding = 0.75
        ):
        return (self.pairwise_similarity(features, ref_features) - flooding).abs()

    def forward_scaffold_hopping(
        self,
        ref_smiles,
        start_smiles = None,
        mode = 'MCMC'
    ):
        ref_features = self.encode([ref_smiles]) 
        if start_smiles is None:
            start_features = self.encoding_profile['Mean'] * torch.randn_like(ref_features).cuda()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles]), 
                self.Generator.get_max_len([ref_smiles]) + 24
            ]
        else:
            start_features = self.encode([start_smiles]).clone().detach()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles, start_smiles]), 
                self.Generator.get_max_len([ref_smiles, start_smiles]) + 24
            ]
        output = self.molecular_generation(
            start_features,
            partial(self.scaffold_hopping, ref_features=ref_features),
            mode = mode,
            length_list = length_list
        )
        output.drop_duplicates(
            subset = ['smiles'], 
            keep = 'first', 
            inplace = True,
            ignore_index = True
        )
        results = self.virtual_screening(
            [ref_smiles], 
            output, 
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = ['Pearson'], 
            worker_num = 1
        )
        del results['features']
        return results
    
    def directional_scaffold_hopping(
            self,
            features,
            ref_features,
            flooding = 0.75
        ):
        pred_labels = self.proptery_scorer(
            features     
        )
        return (
            len(self.Predictor_Info)/(1-flooding)
        ) * (
            self.pairwise_similarity(features, ref_features) - flooding
        ).abs() - torch.sum(
            pred_labels, dim = -1
        )

    def forward_directional_scaffold_hopping(
        self,
        ref_smiles,
        start_smiles = None,
        mode = 'MCMC'
    ):
        ref_features = self.encode([ref_smiles]) 
        if start_smiles is None:
            start_features = self.encoding_profile['Mean'] * torch.randn_like(ref_features).cuda()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles]), 
                self.Generator.get_max_len([ref_smiles]) + 24
            ]
        else:
            start_features = self.encode([start_smiles]).clone().detach()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles, start_smiles]), 
                self.Generator.get_max_len([ref_smiles, start_smiles]) + 24
            ]
        output = self.molecular_generation(
            start_features,
            partial(self.directional_scaffold_hopping, ref_features=ref_features),
            mode = mode,
            length_list = length_list
        )
        output.drop_duplicates(
            subset = ['smiles'], 
            keep = 'first', 
            inplace = True,
            ignore_index = True
        )
        results = self.virtual_screening(
            [ref_smiles], 
            output, 
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = ['Pearson'], 
            worker_num = 1
        )
        del results['features']
        results = self.predict_scores(results, smiles_column = 'smiles')
        return results
    
    def directional_optimization(
            self,
            features,
            ref_features,
            flooding = 0.8
        ):
        pred_labels = self.proptery_scorer(
            features     
        )
        return (
            len(self.Predictor_Info)/(1-flooding)
        ) * (
            self.pairwise_similarity(features, ref_features) - flooding
        ).abs() - torch.sum(
            pred_labels, dim = -1
        )

    def forward_directional_optimization(
        self,
        ref_smiles,
        start_smiles = None,
        mode = 'MCMC'
    ):
        ref_features = self.encode([ref_smiles]) 
        if start_smiles is None:
            start_features = self.encoding_profile['Mean'] * torch.randn_like(ref_features).cuda()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles]), 
                self.Generator.get_max_len([ref_smiles]) + 24
            ]
        else:
            start_features = self.encode([start_smiles]).clone().detach()
            length_list = [ 
                self.Generator.get_max_len([ref_smiles, start_smiles]), 
                self.Generator.get_max_len([ref_smiles, start_smiles]) + 24
            ]
        output = self.molecular_generation(
            start_features,
            partial(self.directional_optimization, ref_features=ref_features),
            mode = mode,
            length_list = length_list
        )
        output.drop_duplicates(
            subset = ['smiles'], 
            keep = 'first', 
            inplace = True,
            ignore_index = True
        )
        results = self.virtual_screening(
            [ref_smiles], 
            output, 
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = ['Pearson'], 
            worker_num = 1
        )
        del results['features']
        results = self.predict_scores(results, smiles_column = 'smiles')
        return results