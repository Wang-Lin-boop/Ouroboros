import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rdkit import Chem

if __name__ == '__main__':
    cxsmiles_file = sys.argv[1]
    output_fn = sys.argv[2]
    table = pd.read_csv(cxsmiles_file, sep='\t', engine='python')
    ps = Chem.SmilesParserParams()
    ps.parseName = False
    table['SMILES'] = table['smiles'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x, ps)))
    table[['SMILES', 'idnumber']].to_csv(f'{output_fn}.csv', header=False, index=False)


