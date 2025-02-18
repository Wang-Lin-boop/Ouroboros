import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.fingerprint import Fingerprint
from utils.chem import gen_standardize_smiles, get_skeleton
from utils.cluster import Mol_Encoder, unsupervised_clustering

if __name__ == '__main__':
    dataset = pd.read_csv(sys.argv[1])
    dataset_name = sys.argv[2]
    smiles_column = sys.argv[3]
    label_column = sys.argv[4]
    test_ratio = float(sys.argv[5])
    val_ratio = float(sys.argv[6])
    method = sys.argv[7]
    if method == "skeleton":
        dataset[smiles_column] = dataset[smiles_column].apply(lambda x:gen_standardize_smiles(x))
        dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
        dataset['skeleton'] = dataset[smiles_column].apply(lambda x:get_skeleton(x))
        dataset = dataset[dataset['skeleton']!='smiles_unvaild']
        skeleton_column = 'skeleton'
    else:
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
            smiles_column = smiles_column,
            method_list = method.split(":")
        )
        dataset, features_columns = predictor.encode(
            dataset
        )
        analyzer = unsupervised_clustering()
        labels = analyzer.cluster_features(
            dataset[features_columns], 
            'Birch', 
            min(200, len(dataset))
        )
        dataset = dataset.join(labels, how='left')
        skeleton_column = 'Birch_ID'
        for feat in features_columns:
            del dataset[feat] 
    dataset_pass = False
    while dataset_pass == False:
        train_skeletons, val_and_test_skeletons = train_test_split(
            dataset[skeleton_column].unique(), 
            test_size=test_ratio
        )
        test_skeletons, val_skeletons = train_test_split(
            val_and_test_skeletons, test_size=val_ratio
        )
        train = pd.DataFrame(columns=dataset.columns)
        test = pd.DataFrame(columns=dataset.columns)
        val = pd.DataFrame(columns=dataset.columns)
        for skeleton in dataset[skeleton_column].unique().tolist():
            if skeleton in train_skeletons:
                train = pd.concat(
                    [train, dataset[dataset[skeleton_column] == skeleton]],
                    ignore_index = True
                )
            elif skeleton in val_skeletons:
                val = pd.concat(
                    [val, dataset[dataset[skeleton_column] == skeleton]], 
                    ignore_index = True
                )
            elif skeleton in test_skeletons:
                test = pd.concat(
                    [test, dataset[dataset[skeleton_column] == skeleton]], 
                    ignore_index = True
                )
        if label_column is None:
            dataset_pass = True
        elif len(set(train[label_column].to_list())) >= 2 and len(set(val[label_column].to_list())) >= 2 and len(set(test[label_column].to_list())) >= 2:
            dataset_pass = True
    print(dataset_name, '::   Train:', len(train), '  Validation:', len(val), '  Test:', len(test))
    train.to_csv(f'{dataset_name}_scaffold_train.csv', index=False)
    val.to_csv(f'{dataset_name}_scaffold_valid.csv', index=False)
    test.to_csv(f'{dataset_name}_scaffold_test.csv', index=False)
    all_data = pd.concat([train, val, test], ignore_index=True)
    all_data.to_csv(f'{dataset_name}_scaffold_all.csv', index=False)