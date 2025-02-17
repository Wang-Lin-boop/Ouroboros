import os
import sys
import torch
import pandas as pd
from model.GeminiMol import GeminiMol
from utils.chem import gen_standardize_smiles, check_smiles_validity, is_valid_smiles
import gc

def aggregate(x):
    return ':'.join(x.unique())

class Pharm_Profiler:
    def __init__(self, 
            encoder, 
            standardize = False
        ):
        self.encoder = encoder
        self.probes_dict = {
            # name : { 
            # 'smiles': smiles_list: 
            # 'weight': weight (float) 
            # }
        }
        self.standardize = standardize

    def prepare(self, dataset, smiles_column='smiles'):
        if self.standardize == True:
            dataset[smiles_column] = dataset[smiles_column].apply(
                lambda x:gen_standardize_smiles(x, kekule=False)
            )
        else:
            dataset[smiles_column] = dataset[smiles_column].apply(
                lambda x:check_smiles_validity(x)
            )
        dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
        return dataset

    def update_probes(self, 
        name,
        smiles_list, 
        weight
    ):
        for smiles in smiles_list:
            assert is_valid_smiles(smiles), f'Error: the probe {smiles} is invalid.'
        self.probes_dict[name] = {
            'smiles': smiles_list,
            'weight': weight
        }

    def update_library(self,
        compound_library,
        prepare = True,
        smiles_column = 'smiles',
    ):
        if prepare:
            compound_library = self.prepare(compound_library, smiles_column=smiles_column)
        print(f'NOTE: the compound library contains {len(compound_library)} compounds.')
        compound_library = compound_library.groupby(smiles_column).agg(aggregate).reset_index()
        print(f'NOTE: non-duplicates compound library contains {len(compound_library)} compounds.')
        self.database = self.encoder.create_database(
            compound_library, 
            smiles_column = smiles_column, 
            worker_num = 1
        )
        print('NOTE: features database was created.')
        gc.collect()
        return self.database

    def __call__(
        self, 
        smiles_column = 'smiles',
        probe_cluster = False,
        smiliarity_metrics = 'Cosine',
        flooding = 0.0
    ):
        print(f'NOTE: columns of feature database: {self.database.columns}')
        res_columns = [x for x in self.database.columns if x != 'features'] 
        total_res = self.database[res_columns]
        gc.collect()
        print(f'NOTE: starting screening...')
        score_list = []
        for name, probe in self.probes_dict.items():
            print(f'NOTE: using {name} as the probe....', end='')
            probe_list = probe['smiles']
            gc.collect()
            if probe_cluster:
                probe_res = self.encoder.virtual_screening(
                    probe_list, 
                    self.database, 
                    input_with_features = True,
                    smiles_column = smiles_column, 
                    return_all_col = False,
                    similarity_metrics = [smiliarity_metrics],
                    worker_num = 4
                )
                probe_res.sort_values(smiliarity_metrics, ascending=False, inplace=True)
                probe_res.drop_duplicates(
                    subset = [smiles_column], 
                    keep = 'first', 
                    inplace = True,
                    ignore_index = True
                )
                probe_res[smiliarity_metrics] = probe_res[smiliarity_metrics].apply(
                    lambda x: x if x > flooding else 0.0
                )
                probe_res[f'{name}'] = probe['weight'] * probe_res[smiliarity_metrics]
                total_res = pd.merge(
                    total_res,
                    probe_res[[smiles_column, f'{name}']], 
                    on = smiles_column
                )
                score_list.append(f'{name}')
            else:
                for i in range(len(probe_list)):
                    probe_res = self.encoder.virtual_screening(
                        [probe['smiles'][i]], 
                        self.database, 
                        input_with_features = True,
                        reverse = False, 
                        smiles_column = smiles_column, 
                        return_all_col = False,
                        similarity_metrics = [smiliarity_metrics],
                        worker_num = 4
                    )
                    probe_res[smiliarity_metrics] = probe_res[smiliarity_metrics].apply(
                        lambda x: x if x > flooding else 0.0
                    )
                    probe_res[f'{name}_{i}'] = probe['weight'] * probe_res[smiliarity_metrics]
                    total_res = pd.merge(
                        total_res,
                        probe_res[[smiles_column, f'{name}_{i}']], 
                        on = smiles_column
                    )
                    score_list.append(f'{name}_{i}')
            print('Done!')
            probe_res = None
            gc.collect()
        total_res['Score'] = total_res[score_list].apply(lambda row: (row - flooding).clip(lower=0).sum(), axis=1)
        return total_res

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    ## load model
    model_name = sys.argv[1]
    encoder = GeminiMol(
            model_name,
            batch_size = 2048
        )
    predictor = Pharm_Profiler(
        encoder,
        standardize = False
        )
    # job_name
    job_name = sys.argv[2]
    smiles_col = sys.argv[3]
    library_path = f'{os.path.dirname(sys.argv[4])}/{os.path.splitext(os.path.basename(sys.argv[4]))[0]}' 
    # update profiles
    ref_smiles_table = pd.read_csv(sys.argv[5])
    if sys.argv[6] not in ['Fasle', 'None', 'F', 'No', 'N', 'no', 'false', 'none']:
        group_param = sys.argv[6].split(':')
        if len(group_param) == 1:
            label_col = group_param[0]
            for group in ref_smiles_table[label_col].to_list(): 
                if isinstance(group, float):
                    predictor.update_probes(
                        name = f'w_{group}',
                        smiles_list = ref_smiles_table[
                            ref_smiles_table[label_col]==group
                        ][smiles_col].to_list(),
                        weight = group
                        )
                else:
                    predictor.update_probes(
                        name = f'{group}',
                        smiles_list = ref_smiles_table[
                            ref_smiles_table[label_col]==group
                        ][smiles_col].to_list(),
                        weight = 1.0
                    )
        elif len(group_param) == 2:
            label_col = group_param[0]
            weight_col = group_param[1]
            for group in ref_smiles_table[label_col].to_list():
                for weight in ref_smiles_table[weight_col].to_list():
                    predictor.update_probes(
                        name = f'{group}_{weight}',
                        smiles_list = ref_smiles_table[
                            ref_smiles_table[label_col]==group & ref_smiles_table[weight_col]== weight
                        ][smiles_col].to_list(),
                        weight = weight
                    )
        else:
            assert len(group_param) <= 2, "Error: profile tags no more than 2!"
    else:
        predictor.update_probes(
            name = 'active',
            smiles_list = ref_smiles_table[smiles_col].to_list(),
            weight = 1.0
        )
    probe_cluster = True if sys.argv[7] in [
        'True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y'
        ] else False
    flooding = float(sys.argv[8])
    keep_ratio = float(sys.argv[9])
    # generate features database
    if os.path.exists(f'{library_path}.parquet'):
        predictor.database = pd.read_parquet(f'{library_path}.parquet')
    else:
        compound_library = pd.read_csv(sys.argv[4])
        features_database = predictor.update_library(
            compound_library,
            prepare = True,
            smiles_column = smiles_col,
        )
        # save database to parquet
        features_database.to_parquet(f'{library_path}.parquet')
        del compound_library
        del features_database
        gc.collect()
    # virtual screening 
    keep_number = int(len(predictor.database)*keep_ratio)
    total_res = predictor(
        smiles_column = smiles_col,
        probe_cluster = probe_cluster,
        flooding = flooding
    )
    total_res = total_res.nlargest(keep_number, 'Score', keep='all')
    total_res.sort_values('Score', ascending=False, inplace=True)
    total_res.to_csv(f"{job_name}_results.csv", index=False, header=True, sep=',')
    print(f'NOTE: job completed! check {job_name}_results.csv for results!')

