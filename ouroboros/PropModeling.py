import sys
import os
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
    ouroboros_path = sys.argv[2]
    smiles_names = sys.argv[3]
    label_info = sys.argv[4]
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
    task_type_list = []
    label_map = {
            'Active': 1, 
            'Inactive': 0, 
            'active': 1, 
            'inactive': 0, 
            1: 1, 
            0: 0
        }
    for label in label_dict.keys():
        train_data[label] = train_data[label].replace(label_map)
        val_data[label] = val_data[label].replace(label_map)
        test_data[label] = test_data[label].replace(label_map)
        max_value = train_data[label].max()
        min_value = train_data[label].min()
        if min_value == 0.0 and max_value == 1.0:
            print(f'NOTE: binary {label}.')
            task_type_list.append('binary')
        else:
            print(f'NOTE: regression {label} max value: {max_value}, min value: {min_value}.')
            train_data[label] = train_data[label].apply(lambda x: (x - min_value) / (max_value - min_value))
            val_data[label] = val_data[label].apply(lambda x: (x - min_value) / (max_value - min_value))
            test_data[label] = test_data[label].apply(lambda x: (x - min_value) / (max_value - min_value))
            task_type_list.append('regression')
    if 'regression' in task_type_list and 'binary' in task_type_list:
        task_type = 'regression'
        print(f'NOTE: some label is regression, and some is binary. So we set {predictor_name} as regression model.')
    elif 'regression' in task_type_list:
        task_type = 'regression'
        print(f'NOTE: all label is regression. So we set {predictor_name} as regression model.')
    else:
        task_type = 'binary'
        print(f'NOTE: all label is binary. So we set {predictor_name} as binary classification model.')
    # set hyper-parameters based on the number of training samples
    if len(train_data) > 4096:
        batch_size, learning_rate, batch_group, hidden_dim = 512, 5.0e-5, 10, 30720
    elif len(train_data) > 2048:
        batch_size, learning_rate, batch_group, hidden_dim = 256, 3.0e-5, 10, 20480
    elif len(train_data) > 512:
        batch_size, learning_rate, batch_group, hidden_dim = 128, 1.0e-5, 10, 12288
    else:
        batch_size, learning_rate, batch_group, hidden_dim = 72, 1.0e-5, 10, 8192
    ## create the encoder models
    QSAR_model = QSAR(
        model_name = ouroboros_path,
        predictor_name = predictor_name,
        batch_size = batch_size,
        predictor_info = {
            'smiles_name': smiles_names,
            'task_type': task_type,
            'label_dict': label_dict,
        },
        params = {
            "hidden_dim": hidden_dim, # between 2048 and 12288
            "dropout_rate": 0.5, 
            "activation": 'Sigmoid',
            "projection_transform": 'Sigmoid',
            "linear_projection": False
        }
    )
    QSAR_model.fit(
        train_data.sample(frac=1.0),
        val_data,
        epochs = (batch_group*2000*batch_size // len(train_data))+1,
        learning_rate = learning_rate,
        optim_type = 'AdamW',
        weight_decay = 0.01,
        patience = 40,
        frozen_steps = 500,
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
        f"{predictor_name}_{os.path.basename(ouroboros_path)}_results.csv", 
        index=True, 
        header=True, 
        sep=','
    )
    print(f"NOTE: {predictor_name} Test Score: {test_score} Val Score: {QSAR_model.best_model_score}")
    print('============================')
