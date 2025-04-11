import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from utils.fingerprint import Fingerprint
import matplotlib.pyplot as plt
from utils.chem import check_smiles_validity, gen_standardize_smiles
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from utils.metrics import cluster_models, reduce_dimension

class Mol_Encoder:
    def __init__(
        self, 
        encoder_list=[Fingerprint(['ECFP4'])], 
        standardize=False, 
        smiles_column='smiles',
        method_list=None
    ):
        self.standardize = standardize
        self.encoders = encoder_list
        self.smiles_column = smiles_column
        self.method_list = method_list

    def prepare(self, dataset):
        dataset = dataset.dropna(subset=[self.smiles_column])
        print(f"NOTE: read the dataset size ({len(dataset)}).")
        if self.standardize == True:
            dataset[self.smiles_column] = dataset[self.smiles_column].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
        else:
            dataset[self.smiles_column] = dataset[self.smiles_column].apply(lambda x: check_smiles_validity(x))
        dataset = dataset[dataset[self.smiles_column]!='smiles_unvaild']
        dataset.drop_duplicates(
            subset=[self.smiles_column], 
            keep='first', 
            inplace=True,
            ignore_index = True
        )
        print(f"NOTE: processed dataset size ({len(dataset)}).")
        dataset.reset_index(drop=True, inplace=True)
        return dataset

    def encode(self, query_smiles_table):
        features_columns = []
        query_smiles_table = self.prepare(query_smiles_table)
        for idx, single_encoder in enumerate(self.encoders):
            features = single_encoder.extract_features(
                query_smiles_table, 
                smiles_column=self.smiles_column,
                prefix=f'{self.method_list[idx]}_'
            ) if self.method_list is not None else single_encoder.extract_features(
                query_smiles_table, 
                smiles_column=self.smiles_column
            ) 
            features_columns += list(features.columns)
            query_smiles_table = query_smiles_table.join(features, how='left')
        return query_smiles_table, features_columns

class unsupervised_clustering:
    def __init__(self):
        self.cluster_models = cluster_models
        self.reduce_dimension = reduce_dimension
    
    def evaluate_clustering(self, y_true, y_pred):
        # purity and rand socre for clustering results
        cm = confusion_matrix(y_true, y_pred) # y_true, y_pred is array-like, shape (n_samples,)
        purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        rand_score = adjusted_rand_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        return purity, rand_score, ari # purity, rand_score, adjusted_rand_score

    def reduce_dimension_plot(self, 
            features, 
            label_series, 
            output_fn=None, 
            method='tSNE', 
            dim_num=2, 
            plot_dim=[0, 1]
        ):
        plt.figure(figsize=(4.5, 4.5), dpi=600)
        if output_fn is None:
            output_fn = f"{self.output_fn}_{self.model_name}_{method}"
        features_array = features.values 
        X_embedded = self.reduce_dimension[method](features_array, dim_num, 1207) # random seed 1207
        point_size = 6 + min(1200/len(label_series), 20)
        if all(isinstance(x, (float, int)) for x in label_series.to_list()):
            if len(label_series) >= 30:
                label_series = label_series.apply(lambda x: round(x, 1))
            label_set = sorted(list(set(label_series.to_list())))
            colors = plt.cm.rainbow(np.linspace(0.15, 0.85, len(label_set)))
        else:
            label_set = [label[0] for label in sorted(label_series.value_counts().to_dict().items(), key=lambda x: x[1], reverse=True)]
            colors = plt.cm.rainbow(np.linspace(0.15, 0.85, len(label_set)))
        color_id = 0
        for label in label_set:
            idx = (label_series.to_numpy() == label).nonzero()
            plt.scatter(X_embedded[idx, plot_dim[0]], X_embedded[idx, plot_dim[1]], c=colors[color_id], label=f'label={label}', marker = '.', s=point_size)
            color_id += 1
        if len(label_set) <= 6:
            plt.legend(loc='best')
        plt.title(f"{output_fn}")
        plt.tight_layout()
        plt.savefig(f"{output_fn}.png")
        plt.close()
        X_embedded = pd.DataFrame(X_embedded, columns=[f'{method}_1', f'{method}_2'])
        return X_embedded.join(label_series, how='left')

    def cluster_features(self, 
            features, 
            algorithm, 
            num_clusters=None
        ):
        features_array = features.values 
        cluster_model = self.cluster_models[algorithm](num_clusters).fit(features_array)
        labels = cluster_model.labels_
        return pd.DataFrame(labels, columns=[f'{algorithm}_ID'])
    
if __name__ == "__main__":
    # random_seed
    random_seed = 1207
    np.random.seed(random_seed)
    # params
    data_table = pd.read_csv(sys.argv[1])
    method = sys.argv[2]
    smiles_column = sys.argv[3]
    ## setup task
    output_fn = sys.argv[4]
    cluster_num = int(sys.argv[5])
    ## read the models
    encoders = {}
    for model_name in method.split(":"):
        if os.path.exists(f'{model_name}/GeminiMol.pt'):
            from model.GeminiMol import GeminiMol
            encoders[model_name] = GeminiMol(
                model_path = model_name, 
                batch_size = 512,
                cache = True
            )
        elif model_name == "CombineFP":
            methods_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
            encoders[model_name] = Fingerprint(methods_list)
        else:
            methods_list = [model_name]
            encoders[model_name] = Fingerprint(methods_list)
    encoders_list = list(encoders.values())
    ## setup encoder
    predictor = Mol_Encoder(
        encoders_list, 
        standardize = True,
        smiles_column = smiles_column
    )
    dataset, features_columns = predictor.encode(data_table)
    ## setup analyzer
    analyzer = unsupervised_clustering()
    for cluster_algorithm in [
        'hierarchical', 
        'K-Means',
        'MeanShift',
        'DBSCAN',
        'AffinityPropagation',
        'Spectral',
        'Birch',
        'OPTICS'
    ]:
        labels = analyzer.cluster_features(
            dataset[features_columns], 
            cluster_algorithm, 
            cluster_num
        )
        dataset = dataset.join(labels, how='left')
    dataset.to_csv(f"{output_fn}_encoding.csv", index=False)
    for feat in features_columns:
        del dataset[feat]
    dataset.to_csv(f"{output_fn}_clusters.csv", index=False, header=True, sep=',')