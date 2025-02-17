import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import (
    initialize_weights,
    activation_dict, 
)

vocabularies = {
"SMILES":{
    '<PAD>': 0, 
    '<CLS>': 1,
    '<EOS>': 2, 
    'C': 3, 
    'O': 4, 
    'N': 5, 
    'S': 6, 
    'P': 7, 
    'F': 8, 
    'Cl': 9, 
    'Br': 10, 
    'I': 11, 
    'c': 12, 
    'o': 13, 
    'n': 14, 
    's': 15, 
    '\\': 16, 
    '/': 17, 
    '=': 18, 
    '#': 19, 
    '-': 20, 
    '+': 21, 
    '@': 22, 
    ']': 23, 
    '[': 24, 
    '(': 25, 
    ')': 26, 
    '1': 27, 
    '2': 28, 
    '3': 29, 
    '4': 30, 
    '5': 31, 
    '6': 32, 
    '7': 33, 
    '8': 34, 
    '9': 35, 
    '0': 36, 
    'p': 37, 
    'B': 38, 
    'Se': 39, 
    'Si': 40, 
    'H': 41, 
    '%': 42,
    }
}

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

def duplicate_interleave(m):
    dim0 = m.shape[0]
    m = m.view(-1, 1)
    m = m.repeat(1, 2)
    m = m.view(dim0, -1)
    return m

class XPOS(nn.Module):
    def __init__(
            self, 
            input_dim, 
            scale_base=512
        ):
        super().__init__()
        self.input_dim = input_dim
        self.scale_base = scale_base # 
        self.register_buffer(
            "scale", (
                torch.arange(0, input_dim, 2) + 0.4 * input_dim
            ) / (1.4 * input_dim)
        )

    def forward(
            self, 
            x, 
            downscale=False,
            min_pos = 0,
            offset = 0
        ):
        batch_size, seq_len, input_dim = x.size()
        max_pos = seq_len + offset + min_pos
        scale = self.scale ** torch.arange(
            min_pos, max_pos, 1
        ).to(self.scale).div(self.scale_base)[:, None]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, scale.size(1)) / scale.size(1)))
        sinusoid_inp = (
            torch.einsum("i , j -> i j", torch.arange(
                0, seq_len, dtype=torch.float).cuda(), inv_freq.cuda()).to(x)
        )
        sin, cos = torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
        if downscale:
            scale = 1 / scale
        sin, cos = map(lambda t: duplicate_interleave(t * scale.cuda()), (sin.cuda(), cos.cuda()))
        x = (x * cos) + (rotate_every_two(x) * sin)
        return x

class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            encoding_size, 
            vocab_dict = None,
            activation = 'GELU',
            num_layers = 4,
            num_heads = 4
    ):
        super(TransformerDecoder, self).__init__()
        self.vocab_dict = vocab_dict
        self.token_dict = {v: k for k, v in vocab_dict.items()}
        self.pad_id = 0
        self.num_heads = num_heads
        self.hidden_dim = encoding_size
        self.num_layers = num_layers
        self.xpos = XPOS(
            input_dim = encoding_size, 
            scale_base = 16
        )
        self.embedding = nn.Embedding(
            len(self.token_dict), 
            encoding_size
        )
        self.cell = torch.compile(
            nn.TransformerDecoder(
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model = encoding_size, 
                    nhead = self.num_heads,
                    batch_first = True,
                    activation = activation_dict['SiLU']
                ), 
                num_layers = num_layers,
                norm = nn.LayerNorm(encoding_size)
            )
        )
        self.output = torch.compile(
            nn.Sequential(
                nn.Linear(encoding_size, encoding_size*2),
                activation_dict[activation],
                nn.Linear(encoding_size*2, 1024),
                activation_dict[activation],
                nn.Linear(1024, len(self.token_dict)),
            )
        )
        self.output.apply(initialize_weights)
        self.cell.apply(initialize_weights)
        self.cell.cuda()
        self.output.cuda()
        self.embedding.cuda()

    def generate_tgt_mask(self, tgt):
        batch_size, tgt_length = tgt.size()
        mask = torch.triu(torch.ones(tgt_length, tgt_length), diagonal=1).bool() # diagonal=1
        tgt_mask = mask.unsqueeze(0).expand(batch_size*self.num_heads, -1, -1)
        return tgt_mask
    
    def sample_indices(self, preds_batch, temperature = 0.10):
        # preds_batch is expected to be of shape [batch_size, seq_length, token_size]
        # Apply softmax with temperature
        log_probs = F.log_softmax(preds_batch / temperature, dim=-1)
        # Sample from the distribution using the Gumbel-Max trick
        u = torch.rand_like(log_probs)
        gumbel_noise = -torch.log(-torch.log(u + 1e-9) + 1e-9)
        sample_log_probs = log_probs + gumbel_noise
        # Take the argmax to get the sampled index
        indices = torch.argmax(sample_log_probs, dim=-1)
        return indices

    def decode(
            self, 
            target_tensor,
            mol_encoding,
            temperature = 0.10,
        ):
        batch_size, tgt_length = target_tensor.size()
        memory = mol_encoding.unsqueeze(1)
        pred = torch.ones((batch_size, 1), dtype=torch.int).cuda()
        # Iterate over the sequence length
        for _ in range(1, tgt_length):
            sequence = self.cell(
                self.xpos(self.embedding(pred)),
                memory,
                tgt_mask = self.generate_tgt_mask(pred).cuda(), 
                tgt_is_causal = True,
                tgt_key_padding_mask = (pred == self.pad_id).cuda(),
            )
            if temperature == 0.0:
                new_tokens = torch.argmax(F.softmax(self.output(sequence[:, -1, :]), dim = -1), dim = -1)
            else:
                new_tokens = self.sample_indices(self.output(sequence[:, -1, :]), temperature)
            pred = torch.cat(
                (
                    pred,
                    new_tokens.unsqueeze(1)
                ), 
                dim = 1
            )
        return pred

class MolecularGenerator(nn.Module):
    def __init__(
            self, 
            encoding_size, 
            vocab_dict = None,            
            chemical_language = 'SMILES',
            params = {
                "num_layers": 4,
                "num_heads": 4,
                "activation": "GELU"
            }
        ):
        super(MolecularGenerator, self).__init__()
        self.chemical_language = chemical_language
        self.vocab_dict = vocab_dict
        self.token_dict = {v: k for k, v in vocab_dict.items()}
        self.pad_id = 0
        self.Decoder = TransformerDecoder(
            encoding_size,
            vocab_dict = vocab_dict,
            activation = params['activation'],
            num_layers = params['num_layers'],
            num_heads = params['num_heads']
        )
        self.max_seq_len = None
        self.Decoder.cuda()
        
    def forward(
            self,
            target_tensor,
            mol_encoding
        ):
        return self.Decoder(
            target_tensor, 
            mol_encoding
        )

    def get_max_len(self, input_sents):
        max_seq_len = max([len(s) for s in input_sents]) + 1 if self.max_seq_len is None else self.max_seq_len
        return max_seq_len

    def indices2smiles(self, indices):
        text = []
        sents = ''
        for token_id in indices:
            if token_id == self.vocab_dict['<CLS>'] or token_id == self.vocab_dict['<PAD>']: # Remove padding tokens
                pass
            elif token_id == self.vocab_dict['<EOS>']:
                break # Stop decoding when the end of sequence token is encountered
            else:
                text.append(self.token_dict[token_id]) 
            # Concatenate the tokens into a single string and add it to the list of texts
            sents = ''.join(text)
        return sents

    def tensor2smiles(self, tensor):
        # tensor: (batch_size, seq_len)
        # vocab: a dictionary that maps token ids to tokens
        # Convert the tensor to a numpy array
        tensor = tensor.detach().cpu().numpy()
        # Iterate over each sequence in the batch
        texts = []
        for indices in tensor:
            # Convert each token id in the sequence to its corresponding token
            smiles = self.indices2smiles(indices)
            texts.append(smiles)
        return texts # [batch_size]

    def decoder(self, mol_encoding, smiles_len=128, temperature = 0.10):
        # gemini_encoding: (batch_size, features_size)
        self.eval()
        with torch.no_grad():
            # Initialize the sequence tensor
            sequence = torch.ones((mol_encoding.size(0), smiles_len), dtype=torch.int).cuda()
            indices = self.Decoder.decode(
                sequence,
                mol_encoding,
                temperature = temperature
            )
            # Find the index of the maximum value along the last dimension
            decoded_text = self.tensor2smiles(indices) 
        return decoded_text