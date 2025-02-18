import sys
import torch
from model.GeminiMol import GeminiMol

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # read data and build dataset
    prefix = sys.argv[1]
    model_name = sys.argv[2]
    # set training params
    training_random_seed = 1207
    torch.manual_seed(training_random_seed)
    torch.cuda.manual_seed(training_random_seed) 
    torch.cuda.manual_seed_all(training_random_seed)
    ouroboros_model = GeminiMol(
        model_path = model_name,
        batch_size = 1024
    )
    ouroboros_model.drawmol(
        [
            
        ],
        prefix = prefix
    )


