import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from model.Ouroboros import Ouroboros
from utils.chem import cal_SAScore, check_smiles_validity

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
    start_smiles = sys.argv[1]
    if check_smiles_validity(start_smiles) != 'smiles_unvaild':
        mol4seed = True
    else:
        mol4seed = False
        start_smiles = None
    model_name = sys.argv[2]
    running_mode = sys.argv[3]
    job_name = sys.argv[4]
    if running_mode.split(":")[0] in [
        'directional_optimization'
    ]:
        predictor_info = {
            'Lipophilicity': 0.6,
            'Caco2': 0.8,
            'Solubility': 0.7,
            'SA': 1.0
        }
        if ":" in running_mode:
            running_params = running_mode.split(":")
            running_mode = running_params[0]
            for prop in running_params[1].split(","):
                predictor_info[prop] = 1.2
        sample_mode = sys.argv[5].split(':')
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
        ouroboros_model = Ouroboros(
            model_name = model_name, 
            batch_size = 128,
            predictor_info = predictor_info,
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
        results = ouroboros_model.directed_exploration(
            start_smiles = start_smiles,
            exploration_obj = running_mode
        )
    elif running_mode == 'stochastic_propagation':
        assert isinstance(start_smiles, str), f'Please provide a SMILES for {running_mode}.'
        ouroboros_model = Ouroboros(
            model_name = model_name, 
            batch_size = 128,
            predictor_info = {},
            driver = None,
            mol4seed = mol4seed,
            driver_params = {}
        )
        results = ouroboros_model.stochastic_propagation(
            start_smiles,
            temperature = 0.1,
            replica_num = 1,
            num_steps_per_replica = 200,
            loud = 0.05
        )
    elif running_mode == 'check':
        assert isinstance(start_smiles, str), f'Please provide a SMILES for {running_mode}.'
        ouroboros_model = Ouroboros(
            model_name = model_name, 
            batch_size = 128,
            predictor_info = {},
            driver = None,
            mol4seed = mol4seed,
            driver_params = {}
        )
        results = ouroboros_model.ouroboros_check(start_smiles)
    else:
        raise Exception(f'Error: {running_mode} is not supported.')
    results['SAScore'] = results['smiles'].apply(cal_SAScore)
    results.to_csv(f'{job_name}_{os.path.basename(model_name)}_generation.csv', index=False, header=True)
