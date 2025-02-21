import os
import sys
import torch
from model.Ouroboros import Ouroboros
import pandas as pd

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    ## load model
    model_name = sys.argv[1]
    encoder = Ouroboros(
        model_name,
        batch_size = 2048,
        predictor_info = {},
        generator = False,
        flooding = [0.3, 0.6], # CSS sim, 2D sim, min sim
        threads = 40
    )
    dataset = pd.read_csv(sys.argv[2].split("@")[0])
    dataset_name = f'{os.path.splitext(os.path.basename(sys.argv[2].split("@")[0]))[0]}'
    smiles_column = sys.argv[2].split("@")[1].split(':')[0]
    label_or_id_column = sys.argv[2].split("@")[1].split(':')[1]
    concise = True if sys.argv[3] in ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y'] else False
    reduce_dimension_method_list = sys.argv[4].split(',')
    dataset = encoder.prepare(
        dataset, smiles_column = smiles_column
    )
    for reduce_dimension_method in reduce_dimension_method_list:
        _ = encoder.reduce_dimension_plot(
            dataset, 
            smiles_column = smiles_column, 
            label_column = label_or_id_column, 
            method = reduce_dimension_method,
            concise = concise,
            output_fn = f'{dataset_name}_{os.path.basename(model_name)}_{reduce_dimension_method}'
        )























