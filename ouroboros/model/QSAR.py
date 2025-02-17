import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .modules import (
    MultiPropDecoder, 
    optimizers_dict, 
    ListLoss
)
from utils.metrics import metric_functions, statistics_dict
from tqdm import tqdm
from .GeminiMol import GeminiMol

class QSAR(GeminiMol):
    def __init__(
            self, 
            model_name = None, 
            predictor_name = 'QSAR',
            batch_size = 512,
            predictor_info = {
                'smiles_name': 'smiles',
                'task_type': 'regression',
                'label_dict': {
                    'score': 'RMSE'
                }
            },
            params = {
                "hidden_dim": 1024,
                "dropout_rate": 0.1,
                "projector": 'KAN',
                "projection_transform": 'Sigmoid',
                "linear_projection": False
            }
        ):
        super().__init__(
            model_path=model_name, 
            batch_size=batch_size,
            cache = True
        )
        self.model_name = model_name
        self.predictor_name = predictor_name
        self.projector = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(
                self.encoder_params['encoding_features'], 
                8192,
            ),
            nn.Sigmoid(),
        )
        self.projector.cuda()
        if os.path.exists(f'{self.model_name}/{self.predictor_name}/Decoder.pt'):
            self.predictor_info = json.load(
                open(
                    f'{self.model_name}/{self.predictor_name}/predictor.json', 
                    'r'
            ))
            params = json.load(
                open(
                    f'{self.model_name}/{self.predictor_name}/model_config.json', 
                    'r'
            ))
            self.label_list = list(self.predictor_info['label_dict'].keys())
            self.decoder = MultiPropDecoder(
                feature_dim = self.encoder_params['encoding_features'] + 8192,
                output_dim = len(self.label_list),
                **params
            )
            self.load()
        else:
            self.predictor_info = predictor_info
            os.makedirs(f'{self.model_name}/{self.predictor_name}', exist_ok=True)
            with open(
                f'{self.model_name}/{self.predictor_name}/predictor.json', 
                'w', 
                encoding='utf-8'
            ) as f:
                json.dump(predictor_info, f, ensure_ascii=False, indent=4)
            with open(
                f'{self.model_name}/{self.predictor_name}/model_config.json', 
                'w', 
                encoding='utf-8'
            ) as f:
                json.dump(params, f, ensure_ascii=False, indent=4)
            self.label_list = list(self.predictor_info['label_dict'].keys())
            self.decoder = MultiPropDecoder(
                feature_dim = self.encoder_params['encoding_features'] + 8192,
                output_dim = len(self.label_list),
                **params
            )
        self.decoder.cuda()

    def load(self):
        if os.path.exists(f'{self.model_name}/{self.predictor_name}/Encoder.pt'):
            self.Encoder.load_state_dict(
                torch.load(f'{self.model_name}/{self.predictor_name}/Encoder.pt')
            )
        if os.path.exists(f'{self.model_name}/{self.predictor_name}/Projector.pt'):
            self.projector.load_state_dict(
                torch.load(f'{self.model_name}/{self.predictor_name}/Projector.pt')
            )
        if os.path.exists(f'{self.model_name}/{self.predictor_name}/Pooling.pt'):
            self.pooling.load_state_dict(
                torch.load(f'{self.model_name}/{self.predictor_name}/Pooling.pt')
            )
        if os.path.exists(f'{self.model_name}/{self.predictor_name}/Decoder.pt'):
            self.decoder.load_state_dict(
                torch.load(f'{self.model_name}/{self.predictor_name}/Decoder.pt')
            )

    def save(self):
        if self.encoder_finetune:
            torch.save(
                self.Encoder.state_dict(), 
                f"{self.model_name}/{self.predictor_name}/Encoder.pt"
            )
            torch.save(
                self.pooling.state_dict(), 
                f"{self.model_name}/{self.predictor_name}/Pooling.pt"
            )
        torch.save(
            self.projector.state_dict(), 
            f"{self.model_name}/{self.predictor_name}/Projector.pt"
        )
        torch.save(
            self.decoder.state_dict(), 
            f"{self.model_name}/{self.predictor_name}/Decoder.pt"
        )

    def forward(self, total_smiles_list):
        # Encode all sentences using the encoder
        features = self.encode(total_smiles_list)
        projected_feats = self.projector(features)
        # Calculate scores of encoded smiles
        return self.decoder(
            torch.cat(
                (projected_feats, features), dim = -1
            )
        )

    def predict_score(
            self, 
            df, 
            as_pandas=True
        ):
        self.eval()
        with torch.no_grad():
            pred_values = {key:[] for key in self.label_list}
            for i in tqdm(range(0, len(df), self.batch_size), desc=f'Predicting {self.predictor_name}'):
                rows = df.iloc[i:i+self.batch_size]
                total_smiles = rows[self.predictor_info['smiles_name']].to_list()
                pred = self.forward(total_smiles).cpu().detach().numpy()
                for k in range(len(self.label_list)):
                    pred_values[self.label_list[k]] += list(pred[:, k])
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values)
                return res_df
            else:
                return pred_values

    def evaluate(
            self, 
            df
        ):
        pred_table = self.predict_score(
            df, 
            as_pandas=True
        )
        model_score = 0
        if len(self.label_list) == 1 and self.label_list[0] not in df.columns:
            assert 'label' in df.columns
            label_col = ['label']
        else:
            label_col = self.label_list
        results = pd.DataFrame(
                index=self.label_list, 
                columns=statistics_dict[self.predictor_info['task_type']]
            )
        for label_name in self.label_list:
            pred_score = pred_table[label_name].tolist()
            true_score = df['label'].tolist() if 'label' in label_col and len(label_col) == 1 else df[label_name].tolist()
            for metric in statistics_dict[self.predictor_info['task_type']]:
                results.loc[[label_name],[metric]] = metric_functions[metric](true_score, pred_score)
            model_score += results[
                self.predictor_info['label_dict'][label_name]
            ].mean()
        print(results)
        return results, model_score / len(self.label_list)

    def fit(
        self,
        train_set,
        val_set,
        epochs = 10,
        learning_rate = 1.0e-4,
        optim_type = 'AdamW',
        weight_decay = 0.01,
        patience = 50,
        frozen_steps = 1000,
        warmup_factor = 0.1, 
        num_warmup_steps = 1000,
        batch_group = 10,
        mini_epoch = 200,
        T_max = 1000, # setup consine LR params, lr = lr_max * 0.5 * (1 + cos( steps / T_max ))
        loss_func = "BCE",
        focal_params = (0.25, 5)
    ):
        # Load the models and optimizers
        if optim_type not in optimizers_dict:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        print(f"NOTE: training the {self.predictor_name} model with {self.model_name} encoder.")
        # Set up the optimizer
        optimizer = optimizers_dict[optim_type](
            [
                {'params': self.decoder.parameters(), 'lr': learning_rate},
                {'params': self.projector.parameters(), 'lr': learning_rate}
            ], 
            lr = learning_rate,
            weight_decay = weight_decay
        )
        self.encoder_finetune = False
        # set up the early stop params
        _, best_model_score = self.evaluate(val_set)
        print(f"NOTE: The initial model score is {round(best_model_score, 4)}.")
        counter = 0
        batch_id = 0
        # Apply warm-up learning rate
        if num_warmup_steps > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor 
        # training
        (alpha, gamma) = focal_params
        start = time.time()
        start_batch_id = batch_id
        if len(self.label_list) == 1 and self.label_list[0] not in train_set.columns:
            assert 'label' in train_set.columns
            label_col = ['label']
        else:
            label_col = self.label_list
        for epoch in range(epochs):
            self.decoder.train()
            self.projector.train()
            train_set = train_set.sample(frac=1)
            for i in range(0, len(train_set), self.batch_size):
                if batch_id == frozen_steps:
                    optimizer.add_param_group(
                        {'params': self.Encoder.parameters(), 'lr': learning_rate}
                    )
                    self.Encoder.train()
                    optimizer.add_param_group(
                        {'params': self.pooling.parameters(), 'lr': learning_rate}
                    )
                    self.pooling.train()
                    self.encoder_finetune = True
                batch_id += 1
                rows = train_set.iloc[i:i+self.batch_size]
                if len(rows) <= 2:
                    continue
                total_smiles = rows[self.predictor_info['smiles_name']].to_list()
                optimizer.zero_grad()
                # Calculate scores between encoded smiles
                pred = self.forward(total_smiles)
                if self.predictor_info['task_type'] == 'binary' and loss_func == "BCE":
                    loss = torch.mean(
                        nn.BCELoss(
                            reduction = 'none',
                        )(
                            pred, 
                            torch.tensor(
                                rows[label_col].values.tolist(), dtype = torch.float
                            ).cuda()
                        ) * torch.tensor(
                            [(1 - rows[label].mean()) for label in label_col]
                        ).cuda()
                    )
                elif self.predictor_info['task_type'] == 'binary' and loss_func == "Focal":
                    bce_loss = nn.BCELoss(
                            reduction = 'none',
                        )(
                            pred, 
                            torch.tensor(
                                rows[label_col].values.tolist(), dtype = torch.float
                            ).cuda()
                        )
                    focal_loss = alpha * (1 - torch.exp(-bce_loss)) ** gamma * bce_loss
                    loss = torch.mean(focal_loss)
                elif self.predictor_info['task_type'] == 'regression':
                    true_m = torch.tensor(rows[label_col].values.tolist()).cuda()
                    mes_loss = nn.MSELoss()(pred, true_m) 
                    loss = mes_loss
                elif self.predictor_info['task_type'] == 'sort':
                    true_m = torch.tensor(rows[label_col].values.tolist()).cuda()
                    true_v = true_m.view(-1, true_m.shape[-1])
                    pred_v = pred.view(-1, pred.shape[-1])
                    assert true_m.shape[-1] == pred.shape[-1]
                    loss = 0.0
                    for i in range(true_m.shape[-1]):
                        loss += ListLoss()(true_v[:, i], pred_v[:, i])
                loss.backward()
                optimizer.step()
                if batch_id % batch_group == 0:
                    print(f"Epoch {epoch+1}, batch {batch_id}, time {round((time.time()-start)/(batch_id-start_batch_id), 2)}, train loss: {loss}")
                    start = time.time()
                    start_batch_id = batch_id
                    # Set up the learning rate scheduler
                    if batch_id < num_warmup_steps:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate * ( warmup_factor + (1 - warmup_factor)* ( batch_id / num_warmup_steps) )
                    elif batch_id >= num_warmup_steps:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate * 0.05 * (1 + 19 * np.cos( (batch_id % T_max)/T_max))
                if batch_id % mini_epoch == 0:
                    # Evaluation on validation set, if provided
                    _, val_model_score = self.evaluate(val_set)
                    self.decoder.train()
                    self.projector.train()
                    if self.encoder_finetune:
                        self.Encoder.train()
                        self.pooling.train()
                    if val_model_score > best_model_score or np.isnan(best_model_score):
                        best_model_score = val_model_score
                        self.save()
                        if counter > 0:
                            counter -= 1
                    else:
                        counter += 1
                    print(f"NOTE: Epoch {epoch+1}, batch {batch_id}, ValScore: {round(val_model_score, 4)}, patience: {patience-counter}")
                    if counter >= patience:
                        print("NOTE: The parameters was converged, stop training!")
                        break
            if counter >= patience:
                break
        _, val_model_score = self.evaluate(val_set)
        print(f"Training over! Evaluating on the validation set: {val_model_score}")
        if val_model_score > best_model_score:
            best_model_score = val_model_score
            self.save()
        print(f"Best Model Score: {best_model_score}")
        self.load()

