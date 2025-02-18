import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from utils.chem import gen_standardize_smiles

if __name__ == '__main__':
    data_file = sys.argv[1]
    output_path = sys.argv[2]
    # dataset = pd.read_csv(
    #     data_file, 
    #     sep='\s+',
    #     header = None,
    #     names = ['ID', 'SMILES'],
    #     dtype={'ID':str, 'SMILES':str}
    # )
    dataset = pd.read_csv(data_file, sep='\s|,|;|\t| ', engine='python')
    dataset.columns = ['SMILES', 'ID']
    dataset['SMILES'] = dataset['SMILES'].apply(lambda x:gen_standardize_smiles(x))
    dataset = dataset[dataset['SMILES']!='smiles_unvaild']
    dataset['SMILES'] = dataset['SMILES'].apply(lambda x: x if '.' not in x else 'smiles_unvaild')
    dataset = dataset[dataset['SMILES']!='smiles_unvaild']
    dataset.to_csv(output_path, index=False, header=False)