import io
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial

activation_dict = {
    'GELU': nn.GELU(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'SiLU': nn.SiLU(),
    'ELU': nn.ELU(),
    'SELU': nn.SELU(),
    'CELU': nn.CELU(),
    'Softplus': nn.Softplus(),
    'Softsign': nn.Softsign(),
    'Hardshrink': nn.Hardshrink(),
    'Hardtanh': nn.Hardtanh(),
    'Hardsigmoid': nn.Hardsigmoid(),
    'LogSigmoid': nn.LogSigmoid(),
    'Softshrink': nn.Softshrink(),
    'PReLU': nn.PReLU(),
    'Softmin': nn.Softmin(dim=-1),
    'Softmax': nn.Softmax(dim=-1),
    'Softmax2d': nn.Softmax2d(),
    'LogSoftmax': nn.LogSoftmax(dim=-1),
    'Sigmoid': nn.Sigmoid(),
    'Identity': nn.Identity(),
    'Tanhshrink': nn.Tanhshrink(),
    'RReLU': nn.RReLU(),
    'Hardswish': nn.Hardswish(),   
    'Mish': nn.Mish(),
}

optimizers_dict = {
    'AdamW': partial(
        torch.optim.AdamW, 
        betas = (0.9, 0.98),
    ),
    'NAdam': partial(
        torch.optim.NAdam, 
        betas = (0.9, 0.98),
    ),
    'L-AdamW': partial(
        torch.optim.AdamW, 
        betas = (0.95, 0.999),
    ),
    'AdamW-B2-D': partial(
        torch.optim.AdamW, 
        betas = (0.9, 0.95),
    ),
    'AdamW-B2-DD': partial(
        torch.optim.AdamW, 
        betas = (0.9, 0.9),
    ),
    'Adam': partial(
        torch.optim.Adam,
        betas = (0.9, 0.98),
    ),
    'SGD': partial(
        torch.optim.SGD,
        momentum=0.8, 
    ),
    'Adagrad': partial(
        torch.optim.Adagrad,
    ),
    'Adadelta': partial(
        torch.optim.Adadelta,
    ),
    'RMSprop': partial(
        torch.optim.RMSprop,
    ),
    'Adamax': partial(
        torch.optim.Adamax,
    ),
    'Rprop': partial(
        torch.optim.Rprop,
    ),
}

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(module.weight)

def params2layerlist(params, layer_num=6):
    params_list = []
    for _ in range(layer_num):
        params_list.append(params) 
    return params_list

def silence_output(func):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout = original_stdout
        return result
    return wrapper
    
class ListLoss:
    def __init__(self):
        pass

    @staticmethod
    def convert_labels_to_prob(labels):
        exp_labels = torch.exp(labels - labels.max())
        probabilities = exp_labels / torch.sum(exp_labels)
        return probabilities

    @staticmethod
    def listnet_loss(scores, labels):
        softmax_scores = F.softmax(scores, dim=0)
        labels = F.softmax(labels, dim=0)
        loss = -torch.sum(labels * torch.log(softmax_scores + 1e-10))
        return loss

    def __call__(self, true_values, predicted_values):
        prob_labels = self.convert_labels_to_prob(true_values)
        return self.listnet_loss(predicted_values, prob_labels)

class BatchNormNd(nn.Module):
    def __init__(self, num_features):
        super(BatchNormNd, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, shape[-1]) 
        x = self.bn(x)
        x = x.view(*shape)
        return x

class MultiPropDecoder(nn.Module):
    def __init__(self, 
        feature_dim = (1024, 12288),
        hidden_dim = 12288,
        dropout_rate = 0.1, 
        output_dim = 1,
        activation = "SiLU",
        projection_transform = 'Sigmoid',
        linear_projection = False
    ):
        super(MultiPropDecoder, self).__init__()
        self.dense_concentrate = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim[0], hidden_dim, bias=True),
            activation_dict[activation],
            BatchNormNd(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.scattered_concentrate = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim[1], feature_dim[1], bias=True),
            activation_dict[activation],
            BatchNormNd(feature_dim[1]),
            nn.Linear(feature_dim[1], hidden_dim, bias=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim[0], 1024, bias=True),
            BatchNormNd(1024),
            activation_dict[activation],
            nn.Linear(1024, 1024, bias=True),
            BatchNormNd(1024),
            activation_dict[activation],
            nn.Linear(1024, 1024, bias=True),
            activation_dict[activation],
            nn.Linear(1024, output_dim, bias=True),
            activation_dict[projection_transform],
            nn.Linear(output_dim, output_dim) if linear_projection else nn.Identity(),
        )
        self.dense_concentrate.cuda()
        self.scattered_concentrate.cuda()
        self.projection.cuda()
    
    def forward(self, features, projected_feats):
        concentrate_features = self.dense_concentrate(features) + self.scattered_concentrate(projected_feats)
        return self.projection(
            torch.cat(
                (
                    concentrate_features, 
                    features
                ), dim = -1
            )
        )



