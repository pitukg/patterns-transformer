import torch
from torch.utils.data import Dataset
import random

# Special tokens
TOKENS = ['<circle>', '<triangle>', '<rectangle>']
TOKEN2ID = {tok: i for i, tok in enumerate(TOKENS)}

class ParityDataset(Dataset):
    def __init__(self, length, seq_len, pre_generate=False, seed=42):
        self.length = length
        self.seq_len = seq_len
        self.pre_generate = pre_generate
        self.seed = seed
        if pre_generate:
            raise NotImplementedError("Pre-generation is not implemented yet.")
        # Use a Random instance for reproducibility
        self._random = random.Random(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # For reproducibility, reseed for each idx
        rnd = random.Random(self.seed + idx)
        # Sample a random sequence of tokens
        tokens = [rnd.choice(TOKENS) for _ in range(self.seq_len)]
        input_ids = torch.tensor([TOKEN2ID[tok] for tok in tokens], dtype=torch.long)
        # Compute parity: label=1 if all tokens appear an even number of times, else 0
        counts = {tok: tokens.count(tok) for tok in TOKENS}
        label = int(all(c % 2 == 0 for c in counts.values()))
        return input_ids, label 