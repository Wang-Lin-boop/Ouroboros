import sys
import numpy as np
import pandas as pd
import torch
from model.QSAR import QSAR

if __name__ == "__main__":
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # random_seed
    training_random_seed = 1207
    np.random.seed(training_random_seed)
    torch.manual_seed(training_random_seed)
    torch.cuda.manual_seed(training_random_seed) 
    torch.cuda.manual_seed_all(training_random_seed)
    # params
    data_path = sys.argv[1]
    geminimol_path = sys.argv[2]
    smiles_names = sys.argv[3]
    task_type = sys.argv[4].split('@')[0]
    label_info = sys.argv[4].split('@')[1]
    predictor_name = sys.argv[5]
    label_dict = {}
    for label in label_info.split(','):
        label_dict[label.split(':')[0]] = label.split(':')[1]
    # read datasets
    train_data = pd.read_csv(f'{data_path}_scaffold_train.csv')
    print(
        f"{predictor_name} Training Set: Number=", 
        len(train_data), 
        )
    val_data = pd.read_csv(f'{data_path}_scaffold_valid.csv')
    print(
        f"{predictor_name} Validation Set: Number=", 
        len(val_data), 
        )
    test_data = pd.read_csv(f'{data_path}_scaffold_test.csv')
    print(
        f"{predictor_name} Test Set: Number=", 
        len(test_data), 
        )
    # normalize data
    if task_type == 'regression':
        for label in label_dict.keys():
            max_value = train_data[label].max()
            min_value = train_data[label].min()
            print(f'NOTE: {label} max value: {max_value}, min value: {min_value}.')
            train_data[label] = train_data[label].apply(lambda x: (x - min_value) / (max_value - min_value))
            val_data[label] = val_data[label].apply(lambda x: (x - min_value) / (max_value - min_value))
            test_data[label] = test_data[label].apply(lambda x: (x - min_value) / (max_value - min_value))
    else:
        for label in label_dict.keys():
            print(f'NOTE: {label} is a binary label.')
            print(f'NOTE: {len(train_data[train_data[label] == 1])} {label} is 1.')
            print(f'NOTE: {len(train_data[train_data[label] == 0])} {label} is 0.')
    ## create the encoder models
    QSAR_model = QSAR(
        model_name = geminimol_path,
        predictor_name = predictor_name,
        batch_size = 512,
        predictor_info = {
            'smiles_name': smiles_names,
            'task_type': task_type,
            'label_dict': label_dict,
        },
        params = {
            "hidden_dim": 1024,
            "dropout_rate": 0.0,
            "projector": 'SiLU',
            "projection_transform": 'Sigmoid',
            "linear_projection": False
        }
    )
    if len(train_data) > 20000:
        batch_size, learning_rate, batch_group = 128, 1.0e-5, 10
    elif len(train_data) > 5000:
        batch_size, learning_rate, batch_group = 96, 1.0e-5, 10
    elif len(train_data) > 1000:
        batch_size, learning_rate, batch_group = 64, 1.0e-5, 5
    else:
        batch_size, learning_rate, batch_group = 48, 1.0e-5, 5
    QSAR_model.fit(
        train_data.sample(frac=1.0),
        val_data,
        epochs = (batch_group*2000*batch_size // len(train_data))+1,
        learning_rate = learning_rate,
        optim_type = 'AdamW',
        weight_decay = 0.01,
        patience = 60,
        frozen_steps = 999999999999999,
        warmup_factor = 0.1, 
        num_warmup_steps = batch_group*200,
        T_max = batch_group*200,
        batch_group = batch_group,
        mini_epoch = batch_group*20,
    )
    ## test models
    print('======== Job Report ========')
    print(f'{predictor_name} Model performance: ')
    test_res, test_score = QSAR_model.evaluate(test_data)
    test_res.to_csv(
        f"{predictor_name}_results.csv", 
        index=True, 
        header=True, 
        sep=','
    )
    print(f"NOTE: {predictor_name} Test Score: {test_score}")
    print('============================')