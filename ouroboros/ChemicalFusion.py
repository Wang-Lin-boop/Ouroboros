import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import pandas as pd
from model.Ouroboros import Ouroboros
from utils.chem import cal_SAScore

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # set training params
    training_random_seed = 1207
    torch.manual_seed(training_random_seed)
    torch.cuda.manual_seed(training_random_seed) 
    torch.cuda.manual_seed_all(training_random_seed)
    # read data and build dataset
    probe_datasets = pd.read_csv(sys.argv[1].split('@')[0])
    fusion_targets = sys.argv[1].split('@')[1].split(':')
    smiles_col = 'SMILES'
    target_col = 'Target'
    id_col = 'ID'
    model_name = sys.argv[2]
    sample_mode = sys.argv[3].split(':')
    job_name = sys.argv[4]
    if len(sample_mode) == 1:
        sample_mode = [
            sample_mode[0], 
            20,
            200,
            1,
            0.1,
            0.4,
            3.0e-5
        ] if sample_mode[0] == 'MCMC' else [
            sample_mode[0], 
            1,
            600,
            1,
            0.05,
            0.4,
            3.0e-5
        ]
    temperature = float(sys.argv[5])
    ouroboros_model = Ouroboros(
        model_name = model_name, 
        batch_size = 128,
        predictor_info = {
            # 'Lipophilicity': 0.5, # 1.5 if not standardization
            # 'Caco2': 0.8, # -2.0 if not standardization
            # 'Solubility': 0.8, # -2.0 if not standardization
        },
        driver = sample_mode[0],
        mol4seed = False if sys.argv[6] in ['False', 'None', 'F', 'No', 'N', 'no', 'false', 'none'] else True,
        driver_params = {
            'replica_num': int(sample_mode[1]),
            'num_steps_per_replica': int(sample_mode[2]),
            'step_interval': int(sample_mode[3]),
            'loud': float(sample_mode[4]),
            'temperature': float(sample_mode[5]),
            'learning_rate': float(sample_mode[6]),
        }
    )
    similarity_matrix = ouroboros_model.similarity_matrix(
        probe_datasets,
        smiles_column = smiles_col,
        label_column = target_col,
        id_column = id_col,
        reference_label = fusion_targets[0],
        query_label = fusion_targets[1],
        output_fn = job_name
    )
    similarity_matrix.to_csv(f'{job_name}_matrix.csv')
    smiles_list_1 = probe_datasets[probe_datasets[target_col]==fusion_targets[0]][smiles_col].tolist()
    smiles_list_2 = probe_datasets[probe_datasets[target_col]==fusion_targets[1]][smiles_col].tolist()
    fusion_temperature = float(sys.argv[7])
    results = ouroboros_model.chemical_fusion(
        ref_smiles = {
            fusion_targets[0]: smiles_list_1,
            fusion_targets[1]: smiles_list_2
        },
        temperature = fusion_temperature
    )
    results.sort_values('Score', ascending=False, inplace=True)
    results['SAScore'] = results['smiles'].apply(cal_SAScore)
    results.to_csv(f'{job_name}_{os.path.basename(model_name)}_generation.csv', index=False, header=True)