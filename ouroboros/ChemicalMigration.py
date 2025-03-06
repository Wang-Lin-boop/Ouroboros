import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
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
    if len(sys.argv[1].split('.')) == 2:
        ref_smiles = sys.argv[1].split('.')[0]
        start_smiles = sys.argv[1].split('.')[1]
    else:
        ref_smiles = sys.argv[1]
        start_smiles = None
    model_name = sys.argv[2]
    running_mode = sys.argv[3]
    sample_mode = sys.argv[4].split(':')
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
    job_name = sys.argv[5]
    mol4seed = True if sys.argv[6] in ['True', 'true', 'Yes', 'yes', '1', 'ON'] else False
    ouroboros_model = Ouroboros(
        model_name = model_name, 
        batch_size = 128,
        predictor_info = {
            'Lipophilicity': 0.5, # 1.5 if not standardization
            'Caco2': 0.8, # -2.0 if not standardization
            'Solubility': 0.8, # -2.0 if not standardization
        },
        driver = sample_mode[0],
        mol4seed = mol4seed,
        driver_params = {
            'replica_num': int(sample_mode[1]),
            'num_steps_per_replica': int(sample_mode[2]),
            'step_interval': int(sample_mode[3]),
            'loud': float(sample_mode[4]),
            'temperature': float(sample_mode[5]),
            'learning_rate': float(sample_mode[6]),
        }
    )
    if running_mode in [
        'migration', 
        'restricted_migration', 
    ]:
        assert isinstance(ref_smiles, str), f'Please provide a SMILES for {running_mode}.'
        results = ouroboros_model.directed_migration(
            ref_smiles = ref_smiles,
            start_smiles = start_smiles,
            migration_obj = running_mode
        )
    else:
        raise Exception(f'Error: {running_mode} is not supported.')
    results.sort_values('Cosine', ascending=False, inplace=True)
    results['SAScore'] = results['smiles'].apply(cal_SAScore)
    results.to_csv(f'{job_name}_{os.path.basename(model_name)}_generation.csv', index=False, header=True)
