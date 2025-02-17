import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    root_mean_squared_error, 
    accuracy_score, 
    f1_score, 
    recall_score, 
    precision_score, 
    mean_absolute_error, 
    average_precision_score
)
from scipy.stats import pearsonr, spearmanr
import oddt.metrics as vsmetrics

metric_functions = {
    'AUROC': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
    'MAP': lambda y_true, y_pred: average_precision_score(y_true, y_pred, average = 'micro'),
    'AUPRC': lambda y_true, y_pred: average_precision_score(y_true, y_pred),
    'BEDROC': lambda y_true, y_pred: vsmetrics.bedroc(y_true, y_pred, alpha=160.9, pos_label=1),
    'EF1%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=1, pos_label=1, kind='fold'),
    'EF0.5%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.5, pos_label=1, kind='fold'),
    'EF0.1%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.1, pos_label=1, kind='fold'),
    'EF0.05%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.05, pos_label=1, kind='fold'),
    'EF0.01%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.01, pos_label=1, kind='fold'),
    'logAUC': lambda y_true, y_pred: vsmetrics.roc_log_auc(y_true, y_pred, pos_label=1, ascending_score=False, log_min=0.001, log_max=1.0),
    'MAE': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
    'RMSE': lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred),
    'SPEARMANR': lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
    'PEARSONR': lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
    'ACC': lambda y_true, y_pred: accuracy_score(y_true, [round(num) for num in y_pred]),
    'specificity': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred], pos_label=0), 
    'precision': lambda y_true, y_pred: precision_score(y_true, [round(num) for num in y_pred]), 
    'recall': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]), 
    'sensitivity': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]), 
    'f1': lambda y_true, y_pred: f1_score(y_true, [round(num) for num in y_pred]), 
}

statistics_dict = {
    'ranking': [
        'AUROC', 
        'BEDROC', 
        'AUPRC',
        'EF1%', 
        'EF0.5%', 
        'EF0.1%', 
        'EF0.05%', 
        'EF0.01%', 
        'logAUC', 
    ], 
    'binary': [
        'AUROC', 
        'AUPRC',
        'ACC', 
        'MAP'
    ],
    'regression': [
        'SPEARMANR', 
        'PEARSONR',
        'RMSE', 
        'MAE'
    ], 
    'sort': [
        'SPEARMANR', 
        'RMSE', 
        'MAE'
    ]
} 

label_map = {
    'Active': 1, 
    'Inactive': 0, 
    'active': 1, 
    'inactive': 0, 
    'Yes': 1, 
    'No': 0, 
    'yes': 1, 
    'no': 0, 
    'True': 1, 
    'False': 0, 
    'true': 1, 
    'false': 0, 
    'Positive': 1, 
    'Negative': 0, 
    'positive': 1, 
    'negative': 0, 
    1: 1, 
    0: 0
}