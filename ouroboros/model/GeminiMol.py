import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dgl import batch, unbatch
from dgllife.utils import atom_type_one_hot, atom_formal_charge, atom_hybridization_one_hot, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic, ConcatFeaturizer, BaseAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.nn import SetTransformerEncoder
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
        graph_library = {},
        threads = 40
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
        self.threads = threads

    def update_graph(self, smiles):
        self.graph_library[smiles] = smiles_to_bigraph(
            smiles, 
            node_featurizer=self.atom_featurizer, 
            edge_featurizer=self.bond_featurizer
        )

    def make_graph(self, smiles):
        if self.cache:
            if smiles not in self.graph_library.keys():
                self.graph_library[smiles] = smiles_to_bigraph(
                    smiles, 
                    node_featurizer=self.atom_featurizer, 
                    edge_featurizer=self.bond_featurizer
                )
            return self.graph_library[smiles]
        else:
            if smiles in self.graph_library.keys():
                return self.graph_library[smiles]
            else:
                return smiles_to_bigraph(
                    smiles, 
                    node_featurizer=self.atom_featurizer,
                    edge_featurizer=self.bond_featurizer
                )

    def update_library(self, smiles_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = []
            for smiles in smiles_list:
                futures.append(executor.submit(self.update_graph, smiles))
            for _ in concurrent.futures.as_completed(futures):
                pass

    def smiles2graph(self, smiles_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            graphs = list(executor.map(self.make_graph, smiles_list))
        input_tensor = batch(graphs).to('cuda')
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
        activation = 'ReLU' if projector == 'SeT' else projector
        gate_nn = nn.Sequential(
            nn.Linear(num_features, num_features * 3),
            nn.BatchNorm1d(num_features * 3),
            activation_dict[activation],
            nn.Linear(num_features * 3, 1024),
            nn.BatchNorm1d(1024),
            activation_dict[activation],
            nn.Linear(1024, 128),
            activation_dict[activation],
            nn.Linear(128, 128),
            activation_dict[activation],
            nn.Linear(128, 128),
            activation_dict[activation],
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        if projector == 'SeT':
            self.attention = SetTransformerEncoder(
                d_model = num_features,
                n_heads = 32,
                d_head = 256,
                d_ff = num_features * 4,
                n_layers = 2,
                block_type = 'sab',
                m = None,
                dropouth = 0.3,
                dropouta = 0.3
            )
            self.attention.to('cuda')
            self.readout = GlobalAttentionPooling(
                gate_nn = torch.compile(gate_nn),
                # Adding feat_nn reduces the representation learning capability, /
                # with a significant performance degradation on the zero-shot task
            )
        else:
            # init the readout and output layers
            self.readout = GlobalAttentionPooling(
                gate_nn = torch.compile(gate_nn),
                # Adding feat_nn reduces the representation learning capability, /
                # with a significant performance degradation on the zero-shot task
            )
        self.readout.cuda()
    
    def forward(self, mol_graph, get_atom_weights = False):
        if self.projector == 'SeT':
            mol_graph.ndata['atom_type'] = self.attention(
                mol_graph, 
                mol_graph.ndata['atom_type']
            )
        encoding, atom_weights = self.readout(
            mol_graph, 
            mol_graph.ndata['atom_type'], 
            get_attention = True
        )
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
        cache = False,
        threads = 40
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
                cache = cache,
                threads = threads
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
                cache = cache,
                threads = threads
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

    '''
    This method is employed to extract pre-trained GeminiMol encodings from raw molecular SMILES representations.
    query_smiles_table: pd.DataFrame
    '''
    def extract_features(self, query_smiles_table, smiles_column='smiles', prefix='GM_'):
        shape_features = self.extract(
            query_smiles_table, 
            smiles_column, 
            as_pandas=False, 
        )
        return pd.DataFrame(shape_features).add_prefix(prefix)