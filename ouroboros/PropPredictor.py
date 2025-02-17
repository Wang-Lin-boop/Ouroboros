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
    # predictor_list = os.listdir(model_name)
    predictor_list = [sys.argv[4]] # [ p for p in predictor_list if os.path.exists(f'{model_name}/{p}/Decoder.pt')]
    ## read the predictor models
    print('NOTE: loading models...')
    predictor = Ouroboros(
        model_name = model_name, 
        batch_size = 512,
        predictor_info = {predictor: 1.0 for predictor in predictor_list},
    )
    ## prepare
    extrnal_data = prepare(extrnal_data, smiles_column = smiles_col, standardize = False)
    print('NOTE: make predictions...')
    predicted_res = predictor.predict_scores(extrnal_data, smiles_column = smiles_col)
    ## output the results
    predicted_res.to_csv(f"{job_name}_predictions.csv", index=False, header=True, sep=',')
    print(f'NOTE: job completed! check {job_name}_predictions.csv for results!')