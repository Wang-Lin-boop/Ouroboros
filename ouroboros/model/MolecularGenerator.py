import torch
import torch.nn as nn
from rdkit import Chem
import torch.nn.functional as F
import numpy as np
from .modules import (
    initialize_weights,
    activation_dict, 
)
import selfies as sf
from collections import Counter
import concurrent.futures

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
        },
    "SELFIES": {
        '<PAD>': 0, 
        '<CLS>': 1,
        '<EOS>': 2,
        '[Branch1]': 3,
        '[S@H1]': 4,
        '[\\-Ring2]': 5,
        '[/-Ring1]': 6,
        '[C@]': 7,
        '[P]': 8,
        '[Si@@]': 9,
        '[=S@]': 10,
        '[NH1]': 11,
        '[\\N+1]': 12,
        '[/C@H1]': 13,
        '[=Branch1]': 14,
        '[\\C]': 15,
        '[/S]': 16,
        '[=N-1]': 17,
        '[S@@]': 18,
        '[/P]': 19,
        '[=S@@H1]': 20,
        '[=NH1+1]': 21,
        '[\\C@]': 22,
        '[\\H]': 23,
        '[#C]': 24,
        '[C]': 25,
        '[/C@@H1]': 26,
        '[/C@]': 27,
        '[Mg]': 28,
        '[/N]': 29,
        '[\\N-1]': 30,
        '[N@@H1+1]': 31,
        '[#Branch2]': 32,
        '[P@]': 33,
        '[=N]': 34,
        '[S@@H1]': 35,
        '[=S@H1]': 36,
        '[/N-1]': 37,
        '[CH1-1]': 38,
        '[\\Cl]': 39,
        '[\\F]': 40,
        '[=SH1]': 41,
        '[N@@+1]': 42,
        '[SeH1]': 43,
        '[\\S]': 44,
        '[B]': 45,
        '[CH1+1]': 46,
        '[SH1]': 47,
        '[=CH1+1]': 48,
        '[\\P@H1]': 49,
        '[BH1-1]': 50,
        '[/O-1]': 51,
        '[#N+1]': 52,
        '[Cl]': 53,
        '[\\/Ring1]': 54,
        '[O]': 55,
        '[NH2+1]': 56,
        '[\\N]': 57,
        '[/N+1]': 58,
        '[-/Ring1]': 59,
        '[\\O-1]': 60,
        '[O-1]': 61,
        '[OH1-1]': 62,
        '[Ring2]': 63,
        '[/-Ring2]': 64,
        '[\\O]': 65,
        '[\\C@@]': 66,
        '[-\\Ring2]': 67,
        '[C@@]': 68,
        '[N@H1+1]': 69,
        '[I]': 70,
        '[P@@H1]': 71,
        '[\\S-1]': 72,
        '[=NH2+1]': 73,
        '[N+1]': 74,
        '[-/Ring2]': 75,
        '[O+1]': 76,
        '[=SH1+1]': 77,
        '[As]': 78,
        '[=N+1]': 79,
        '[C+1]': 80,
        '[/C]': 81,
        '[\\C@@H1]': 82,
        '[=OH1+1]': 83,
        '[P@@+1]': 84,
        '[N@+1]': 85,
        '[-\\Ring1]': 86,
        '[/I]': 87,
        '[P@H1]': 88,
        '[Branch2]': 89,
        '[P@@]': 90,
        '[\\NH2+1]': 91,
        '[/C+1]': 92,
        '[Si]': 93,
        '[S+1]': 94,
        '[\\C@H1]': 95,
        '[AsH1]': 96,
        '[=Ring2]': 97,
        '[=P@@]': 98,
        '[=Ring1]': 99,
        '[Se]': 100,
        '[MgH1]': 101,
        '[NH3+1]': 102,
        '[I+1]': 103,
        '[=P]': 104,
        '[P+1]': 105,
        '[\\-Ring1]': 106,
        '[#N]': 107,
        '[/S@]': 108,
        '[=C]': 109,
        '[S-1]': 110,
        '[NH4+1]': 111,
        '[=S@@]': 112,
        '[H]': 113,
        '[Te]': 114,
        '[2H]': 115,
        '[\\NH1-1]': 116,
        '[\\NH1+1]': 117,
        '[CH2-1]': 118,
        '[Si@]': 119,
        '[N-1]': 120,
        '[\\P@@]': 121,
        '[/NH1+1]': 122,
        '[S]': 123,
        '[=Branch2]': 124,
        '[/N@@+1]': 125,
        '[#Branch1]': 126,
        '[=S]': 127,
        '[/O]': 128,
        '[/\\Ring1]': 129,
        '[S@]': 130,
        '[F]': 131,
        '[BH3-1]': 132,
        '[C@@H1]': 133,
        '[C@H1]': 134,
        '[N]': 135,
        '[/NH2+1]': 136,
        '[=O]': 137,
        '[NH1-1]': 138,
        '[\\NH1]': 139,
        '[B-1]': 140,
        '[/NH3+1]': 141,
        '[Ring1]': 142,
        '[=S+1]': 143,
        '[C-1]': 144,
        '[/H]': 145,
        '[Br]': 146,
        '[11CH3]': 147,
        '[/S@@]': 148,
        '[NH1+1]': 149,
        '[=P@]': 150,
        '[/Cl]': 151,
        '[/S+1]': 152,
        '[/P@]': 153,
        '[\\Br]': 154,
        '[/Br]': 155,
        '[/C@@]': 156,
        '[/NH1]': 157,
        '[/NH1-1]': 158,
        '[PH1]': 159,
        '[/S-1]': 160,
        '[/B-1]': 161,
        '[/F]': 162,
        '[=O+1]': 163,
        '[/N@@H1+1]': 164,
        '[=PH1]': 165,
        '[=As]': 166
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
            scale_base = 16
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
            num_heads = 4,
            scale_base = 16
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
            scale_base = scale_base
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
        self.embedding.cuda()
        self.cell.cuda()
        self.output.cuda()

    def generate_tgt_mask(self, tgt):
        batch_size, tgt_length = tgt.size()
        mask = torch.triu(torch.ones(tgt_length, tgt_length), diagonal=1).bool() # diagonal=1
        tgt_mask = mask.unsqueeze(0).expand(batch_size*self.num_heads, -1, -1)
        return tgt_mask

    def forward(
        self, 
        target_tensor,
        mol_encoding, # (batch_size, input_dim)
    ):
        memory = mol_encoding.unsqueeze(1)
        # Initialize the sequence tensor
        padding_mask = (target_tensor == self.pad_id).cuda()
        mask = self.generate_tgt_mask(target_tensor).cuda()
        target_tensor = self.xpos(self.embedding(target_tensor))
        # Iterate over the sequence length
        sequence = self.cell(
            target_tensor, 
            memory, 
            tgt_mask = mask, 
            tgt_is_causal = True,
            tgt_key_padding_mask = padding_mask,
        ) # 
        return self.output(sequence)[:, :-1, :]

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
            chemical_language = 'SELFIES',
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

    def tokenizer(self, input_string, max_seq_len = 128):
        # Convert input sentence to a tensor of numerical indices
        indices = [self.vocab_dict['<CLS>']] # [CLS] token
        i = 0
        while i < len(input_string):
            # Find longest matching word in vocabulary
            best_match = None
            for word, index in self.vocab_dict.items():
                if input_string.startswith(word, i):
                    if not best_match or len(word) > len(best_match[0]):
                        best_match = (word, index)
            if best_match:
                indices.append(best_match[1])
                i += len(best_match[0])
            else:
                if indices[-1] != self.vocab_dict['<PAD>']:
                    indices.append(self.vocab_dict['<PAD>']) # No matching word found, use character index (<PAD>)
                i += 1
            if len(indices) == max_seq_len:
                break
        pad_len = max_seq_len - len(indices)
        indices += [self.vocab_dict['<EOS>']] # <EOS> token
        indices += [self.vocab_dict['<PAD>']] * pad_len # <PAD>
        # Reshape indices batch to a rectangular shape, with shape (batch_size, seq_len)
        return indices

    def generate_weights(self, indices_list):
        counter = Counter(np.array(indices_list).flatten().tolist())
        total_tokens = sum(counter.values())
        weights = {
            index: (total_tokens / (token_count or 1)) if (total_tokens / (token_count or 1)) < 100 else 100
            for index, token_count in counter.items()
        }
        return weights

    def smiles2tensor(self, sents, return_weights=False):
        if self.chemical_language == 'SMILES':
            # Get the max sequence length in current batch
            max_seq_len = max([len(s) for s in sents]) if self.max_seq_len is None else self.max_seq_len - 1
        elif self.chemical_language == 'SELFIES':
            sents = list(map(
                sf.encoder, 
                sents
            ))
            max_seq_len = max(sf.len_selfies(s) for s in sents) if self.max_seq_len is None else self.max_seq_len - 1 
        # Create input tensor of shape (batch_size, seq_len)
        # indices = [self.tokenizer(s, max_seq_len) for s in sents]
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            indices = list(executor.map(self.tokenizer, sents, [max_seq_len]*len(sents)))
        input_tensor = torch.cat(
            [torch.tensor(indice).unsqueeze(0).cuda() for indice in indices] , dim=0
        ) # (batch_size, seq_len)
        if return_weights:
            weights = self.generate_weights(indices)
            return input_tensor, weights
        else:
            return input_tensor

    def get_max_len(self, input_sents):
        if self.chemical_language == 'SMILES':
            max_seq_len = max([len(s) for s in input_sents]) + 1 if self.max_seq_len is None else self.max_seq_len
        elif self.chemical_language == 'SELFIES':
            selfies_dataset = list(map(
                sf.encoder, 
                input_sents
            ))
            max_seq_len = max(sf.len_selfies(s) for s in selfies_dataset) + 1 if self.max_seq_len is None else self.max_seq_len
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
        if self.chemical_language in ['smiles', 'SMILES']:
            return sents
        elif self.chemical_language in ['SELFIES', 'selfies']:
            smiles = sf.decoder(sents)
            try:
                mol = Chem.MolFromSmiles(smiles)
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol, kekuleSmiles=False, doRandom=False, isomericSmiles=True)
            except:
                return smiles

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