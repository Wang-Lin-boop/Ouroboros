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
    if len(sys.argv[1].split('.')) == 1:
        ref_smiles = sys.argv[1]
        start_smiles = None
    else:
        ref_smiles = sys.argv[1].split('.')[0]
        start_smiles = sys.argv[1].split('.')[1]
    model_name = sys.argv[2]
    ouroboros_model = Ouroboros(
        model_name = model_name, 
        batch_size = 128,
        predictor_info = {
            'Caco2_Wang': 0.1, 
            'Solubility_AqSolDB': 0.2
        },
    )
    running_mode = sys.argv[3]
    sample_mode = sys.argv[4]
    job_name = sys.argv[5]
    if running_mode == 'migration':
        results = ouroboros_model.forward_migration(
            ref_smiles = ref_smiles,
            start_smiles = start_smiles,
            mode = sample_mode
        )
    elif running_mode == 'directional_optimization':
        results = ouroboros_model.forward_directional_optimization(
            ref_smiles = ref_smiles,
            start_smiles = start_smiles,
            mode = sample_mode
        )
    elif running_mode == 'directional_scaffold_hopping':
        results = ouroboros_model.forward_directional_scaffold_hopping(
            ref_smiles = ref_smiles,
            start_smiles = start_smiles,
            mode = sample_mode
        )
    elif running_mode == 'scaffold_hopping':
        results = ouroboros_model.forward_scaffold_hopping(
            ref_smiles = ref_smiles,
            start_smiles = start_smiles,
            mode = sample_mode
        )
    elif running_mode == 'random_walking':
        results = ouroboros_model.random_walking(
            ref_smiles,
            temperature = 0.1,
            replica_num = 1,
            num_steps_per_replica = 200,
            loud = 0.1
        )
    results.sort_values('Pearson', ascending=False, inplace=True)
    results['SAScore'] = results['smiles'].apply(cal_SAScore)
    results = results[results['SAScore'] >= 0.6]
    results.to_csv(f'{job_name}_generation.csv', index=False, header=True)
