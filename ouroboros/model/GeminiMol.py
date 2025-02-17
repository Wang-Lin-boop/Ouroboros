import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dgl import batch, unbatch
from dgllife.utils import atom_type_one_hot, atom_formal_charge, atom_hybridization_one_hot, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic, ConcatFeaturizer, BaseAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from dgllife.model.gnn.wln import WLN
from .modules import (
    activation_dict
)
import concurrent.futures
from tqdm import tqdm
import os
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm

class MolecularEncoder(nn.Module):
    '''
    MolecularEncoder

    This is a graph encoder model consisting of a graph neural network from 
    DGL and the MLP architecture in pytorch, which builds an understanding 
    of a compound's conformational space by supervised learning of CSS data.

    '''
    def __init__(
        self,
        num_features = 1024,
        num_layers = 4,
        cache = False,
        graph_library = {}
    ):
        super().__init__()
        
        ## set up featurizer
        self.atom_featurizer = BaseAtomFeaturizer({
            'atom_type':ConcatFeaturizer(
                [
                    atom_type_one_hot, 
                    atom_hybridization_one_hot, 
                    atom_formal_charge, 
                    atom_chiral_tag_one_hot, 
                    atom_is_in_ring, 
                    atom_is_aromatic
        ])}) 
        self.bond_featurizer = CanonicalBondFeaturizer(
            bond_data_field='bond_type'
        )
        # init the OBEncoder
        self.OBEncoder = WLN(
            self.atom_featurizer.feat_size(feat_name='atom_type'), 
            self.bond_featurizer.feat_size(feat_name='bond_type'), 
            n_layers = num_layers, 
            node_out_feats = num_features
        )
        self.OBEncoder.to('cuda')
        self.cache = cache
        self.graph_library = graph_library
        self.graph_dict = self.graph_library

    def make_graph(self, smiles):
        if smiles not in self.graph_dict.keys():
            self.graph_dict[smiles] = smiles_to_bigraph(
                smiles, 
                node_featurizer=self.atom_featurizer, 
                edge_featurizer=self.bond_featurizer
            )

    def update_library(self, smiles_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=23) as executor:
            futures = []
            for smiles in smiles_list:
                    futures.append(executor.submit(self.make_graph, smiles))
            for _ in concurrent.futures.as_completed(futures):
                pass
        self.graph_library = self.graph_dict

    def smiles2graph(self, input_sents):
        with concurrent.futures.ThreadPoolExecutor(max_workers=23) as executor:
            futures = []
            for smiles in input_sents:
                    futures.append(executor.submit(self.make_graph, smiles))
            for _ in concurrent.futures.as_completed(futures):
                pass
        input_tensor = batch([self.graph_dict[smiles] for smiles in input_sents]).to('cuda')
        if not self.cache:
            self.graph_dict = self.graph_library
        return input_tensor

    def forward(self, smiles_list):
        mol_graph = self.smiles2graph(smiles_list)
        mol_graph.ndata['atom_type'] = self.OBEncoder(
            mol_graph, 
            mol_graph.ndata['atom_type'], 
            mol_graph.edata['bond_type']
        )
        return mol_graph

class MolecularPooling(nn.Module):
    def __init__(
        self,
        num_features = 512,
        projector = 'LeakyReLU',
    ):
        super(MolecularPooling, self).__init__()
        self.projector = projector
        if projector == 'Mean':
            self.readout = MLPNodeReadout(
                num_features, num_features, num_features,
                activation=activation_dict['Sigmoid'],
                mode='mean'
            )
        else:
            gate_nn = nn.Sequential(
                nn.Linear(num_features, num_features * 3),
                nn.BatchNorm1d(num_features * 3),
                activation_dict[projector],
                nn.Linear(num_features * 3, 1024),
                nn.BatchNorm1d(1024),
                activation_dict[projector],
                nn.Linear(1024, 128),
                activation_dict[projector],
                nn.Linear(128, 128),
                activation_dict[projector],
                nn.Linear(128, 128),
                activation_dict[projector],
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            # init the readout and output layers
            self.readout = GlobalAttentionPooling(
                gate_nn = torch.compile(gate_nn),
                # Adding feat_nn reduces the representation learning capability, /
                # with a significant performance degradation on the zero-shot task
            )
        self.readout.cuda()
    
    def forward(self, mol_graph, get_atom_weights = False):
        if self.projector != 'Mean':
            encoding, atom_weights = self.readout(
                mol_graph, 
                mol_graph.ndata['atom_type'], 
                get_attention = True
            )
        else:
            encoding = self.readout(mol_graph, mol_graph.ndata['atom_type'])
            atom_weights = None
        if get_atom_weights:
            return encoding, atom_weights
        else:
            return encoding

class GeminiMol(nn.Module):
    def __init__(
        self,
        model_path = None,
        batch_size = 512,
        encoder_params = {
            "num_layers": 4,
            "encoding_features": 2048,
        },
        pooling_params = {
            "projector": "LeakyReLU",
        },
        cache = False
    ):
        # basic setting
        torch.set_float32_matmul_precision('high')
        super(GeminiMol, self).__init__()
        ## load parameters
        self.batch_size = batch_size 
        self.model_path = model_path
        self.model_name = model_path
        if model_path is not None and os.path.exists(f'{model_path}/MolEncoder.pt'):
            self.encoder_params = json.load(open(f'{model_path}/encoder_config.json', 'r'))
            ## create MolecularEncoder
            self.Encoder = MolecularEncoder(
                num_layers = self.encoder_params['num_layers'],
                num_features = self.encoder_params['encoding_features'],
                cache = cache
            )
            self.Encoder.load_state_dict(torch.load(f'{model_path}/MolEncoder.pt'))
        else:
            self.encoder_params = encoder_params
            os.makedirs(model_path, exist_ok = True)
            with open(f'{model_path}/encoder_config.json', 'w', encoding='utf-8') as f:
                json.dump(encoder_params, f, ensure_ascii=False, indent=4)
            ## create MolecularEncoder
            self.Encoder = MolecularEncoder(
                num_layers = encoder_params['num_layers'],
                num_features = encoder_params['encoding_features'],
                cache = cache
            )
        if os.path.exists(f'{model_path}/Pooling.pt'):
            self.pooling_params = json.load(open(f'{model_path}/pooling_config.json', 'r'))
            self.pooling = MolecularPooling(
                num_features = self.encoder_params['encoding_features'],
                projector = self.pooling_params['projector'],
            )
            self.pooling.load_state_dict(torch.load(f'{model_path}/Pooling.pt'))
        else:
            self.pooling_params = pooling_params
            with open(f'{model_path}/pooling_config.json', 'w', encoding='utf-8') as f:
                json.dump(pooling_params, f, ensure_ascii=False, indent=4)
            self.pooling = MolecularPooling(
                num_features = encoder_params['encoding_features'],
                projector = pooling_params['projector'],
            )

    def encode(self, smiles_list):
        # Encode all sentences using the encoder 
        mol_graph = self.Encoder(smiles_list)
        features = self.pooling(mol_graph)
        return features
    
    def drawmol(self, smiles_list, prefix = 'Mol_'):
        mol_graph = self.Encoder.smiles2graph(smiles_list)
        mol_graph.ndata['atom_type'] = self.Encoder.OBEncoder(
            mol_graph, 
            mol_graph.ndata['atom_type'], 
            mol_graph.edata['bond_type']
        )        
        mol_graphs = unbatch(mol_graph)
        mol_no = 1
        for smiles, mol_g in zip(smiles_list, mol_graphs):
            fig_name = f'{prefix}.png' if len(smiles_list) == 1 else f'{prefix}_{mol_no}.png'
            mol_no += 1
            _, atom_weights = self.pooling(batch([mol_g]), get_atom_weights = True)
            atom_weights = atom_weights / torch.max(atom_weights)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
            cmap = cm.get_cmap('gray')
            plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
            mol = Chem.MolFromSmiles(smiles)
            rdDepictor.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(380, 380)
            drawer.SetFontSize(1)
            drawer.drawOptions().bondLineWidth = 5
            mol = rdMolDraw2D.PrepareMolForDrawing(mol)
            drawer.DrawMolecule(mol, 
                highlightAtoms=range(mol_g.number_of_nodes()),
                highlightBonds=[],
                highlightAtomColors={
                    i: plt_colors.to_rgba(atom_weights[i].data.item()) for i in range(mol_g.number_of_nodes())
                }
            )
            drawer.FinishDrawing()
            drawer.WriteDrawingText(fig_name)

    def decode(
            self, 
            features, 
            label_name = 'Pearson'
        ):
        # re-shape encoded features
        batch_dim = features.shape[0]//2
        encoded1, encoded2 = features[:batch_dim], features[batch_dim:]
        # decode vector similarity
        if label_name == 'Cosine':
            encoded1_normalized = torch.nn.functional.normalize(encoded1, dim=-1)
            encoded2_normalized = torch.nn.functional.normalize(encoded2, dim=-1)
            return torch.nn.functional.cosine_similarity(encoded1_normalized, encoded2_normalized, dim=-1)
        elif label_name == 'eCosine':
            return torch.nn.functional.cosine_similarity(encoded1, encoded2, dim=-1)
        elif label_name == 'Euclidean':
            return 1 - torch.sqrt(torch.sum((encoded1 - encoded2)**2, dim=-1))
        elif label_name == 'Manhattan':
            return 1 - torch.sum(torch.abs(encoded1 - encoded2), dim=-1)
        elif label_name == 'Minkowski':
            return 1 - torch.norm(encoded1 - encoded2, p=3, dim=-1)
        elif label_name == 'Pearson':
            mean1 = torch.mean(encoded1, dim=-1, keepdim=True)
            mean2 = torch.mean(encoded2, dim=-1, keepdim=True)
            std1 = torch.std(encoded1, dim=-1, keepdim=True)
            std2 = torch.std(encoded2, dim=-1, keepdim=True)
            encoded1_normalized = (encoded1 - mean1) / (std1 + 1e-7)
            encoded2_normalized = (encoded2 - mean2) / (std2 + 1e-7)
            return torch.mean(encoded1_normalized * encoded2_normalized, dim=-1)
        else:
            raise ValueError("Error: unkown label name: %s" % label_name)

    def extract(
        self, 
        df, 
        col_name, 
        as_pandas=True
    ):
        self.eval()
        data = df[col_name].tolist()
        with torch.no_grad():
            features_list = []
            for i in tqdm(range(0, len(data), self.batch_size), desc = 'Extracting'):
                smiles_list = data[i:i+self.batch_size]
                features = self.encode(smiles_list)
                features_list += list(features.cpu().detach().numpy())
        if as_pandas == True:
            return pd.DataFrame(features_list).add_prefix('GM_')
        else:
            return features_list

    def pairwise_similarity(
            self,
            features,
            ref_features,
    ):
        return torch.nn.functional.cosine_similarity(features, ref_features, dim=-1)

    def predict_similarity(
            self, 
            smiles1_list, 
            smiles2_list,
            as_pandas = True, 
            similarity_metrics = ['Pearson']
            ):
        self.eval()
        assert len(smiles1_list) == len(smiles2_list), f'Error: smiles list must be same length.'
        with torch.no_grad():
            pred_values = {key:[] for key in similarity_metrics}
            for i in range(0, len(smiles1_list), self.batch_size):
                sent1 = smiles1_list[i:i+self.batch_size]
                sent2 = smiles2_list[i:i+self.batch_size]
                # Concatenate input sentences
                input_sents = sent1 + sent2
                features = self.encode(input_sents)
                for label_name in similarity_metrics:
                    pred = self.decode(features, label_name)
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values)
                return res_df
            else:
                return pred_values

    def create_database(
            self, 
            query_smiles_table, 
            smiles_column='smiles', 
            worker_num=1
        ):
        data = query_smiles_table[smiles_column].tolist()
        query_smiles_table['features'] = None
        with torch.no_grad():
            for i in tqdm(range(0, len(data), 2*worker_num*self.batch_size), desc = 'Encoding'):
                smiles_list = data[i:i+2*worker_num*self.batch_size]
                features = self.encode(smiles_list)
                features_list = list(features.cpu().detach().numpy())
                for j in range(len(features_list)):
                    query_smiles_table.at[i+j, 'features'] = features_list[j]
        return query_smiles_table

    def database_screening(
            self, 
            shape_database, 
            ref_smiles, 
            as_pandas = True, 
            similarity_metrics=['Pearson'], 
            worker_num=1
        ):
        torch.set_float32_matmul_precision('high')
        features_list = shape_database['features'].tolist()
        with torch.no_grad():
            pred_values = {key:[] for key in similarity_metrics}
            ref_features = self.encode([ref_smiles]).cuda()
            for i in range(0, len(features_list), worker_num*self.batch_size):
                features_batch = features_list[i:i+worker_num*self.batch_size]
                query_features = torch.from_numpy(np.array(features_batch)).cuda()
                features = torch.cat((
                    query_features, 
                    ref_features.repeat(len(features_batch), 1)
                ), dim=0)
                for label_name in similarity_metrics:
                    pred = self.decode(features, label_name)
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=similarity_metrics)
                return res_df
            else:
                return pred_values 

    def virtual_screening(
            self, 
            ref_smiles_list, 
            query_smiles_table, 
            input_with_features = False,
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = None, 
            worker_num = 1
        ):
        if input_with_features:
            features_database = query_smiles_table
        else:
            features_database = self.create_database(
                query_smiles_table, 
                smiles_column = smiles_column, 
                worker_num = worker_num
            )
        total_res = pd.DataFrame()
        for ref_smiles in ref_smiles_list:
            query_scores = self.database_screening(
                features_database, 
                ref_smiles, 
                as_pandas=True, 
                similarity_metrics=similarity_metrics
            )
            assert len(query_scores) == len(query_smiles_table), f"Error: different length between original dataframe with predicted scores! {ref_smiles}"
            if return_all_col:
                total_res = pd.concat([total_res, query_smiles_table.join(query_scores, how='left')], ignore_index=True)
            else:
                total_res = pd.concat([total_res, query_smiles_table[[smiles_column]].join(query_scores, how='left')], ignore_index=True)
        return total_res

    '''
    This method is employed to extract pre-trained GeminiMol encodings from raw molecular SMILES representations.
    query_smiles_table: pd.DataFrame
    '''
    def extract_features(self, query_smiles_table, smiles_column='smiles'):
        shape_features = self.extract(
            query_smiles_table, 
            smiles_column, 
            as_pandas=False, 
        )
        return pd.DataFrame(shape_features).add_prefix('GM_')