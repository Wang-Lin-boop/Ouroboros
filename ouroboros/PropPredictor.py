import os
import sys
import pandas as pd
from model.Ouroboros import Ouroboros
from utils.chem import prepare

if __name__ == "__main__":
    model_name = sys.argv[1]
    smiles_col = sys.argv[2]
    extrnal_data = pd.read_csv(sys.argv[3])
    job_name = f'{os.path.splitext(os.path.basename(sys.argv[3]))[0]}_{os.path.basename(sys.argv[1])}'
    extrnal_data = prepare(extrnal_data, smiles_column = smiles_col, standardize = False)
    print('NOTE: loading models...')
    for predictor_name_str in [ p for p in os.listdir(model_name) if os.path.exists(f'{model_name}/{p}/Decoder.pt')]:
        ## read the predictor models
        predictor = Ouroboros(
            model_name = model_name, 
            batch_size = 512,
            predictor_info = {predictor_name_str: 1.0},
            generator = False
        )
        ## prepare
        print('NOTE: make predictions...')
        extrnal_data = predictor.predict_scores(extrnal_data, smiles_column = smiles_col)
    ## output the results
    extrnal_data.to_csv(f"{job_name}_predictions.csv", index=False, header=True, sep=',')
    print(f'NOTE: job completed! check {job_name}_predictions.csv for results!')