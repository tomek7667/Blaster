import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BacteriaDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.classes = list(set(item[0] for item in data))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        self.sequences = [item[1] for item in data]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        class_name, sequence = self.data[idx]
        class_idx = self.class_to_idx[class_name]
        sequence_one_hot = self.sequence_to_one_hot(sequence)
        return torch.FloatTensor(sequence_one_hot), torch.tensor(class_idx)
    
    def __str__(self):
        return f"BacteriaDataset with {len(self)} items, {len(self.classes)} classes"

    def sequence_to_one_hot(self, sequence):
        nucleotide_to_index = {'G': 0, 'A': 1, 'T': 2, 'C': 3}
        one_hot_sequence = np.zeros((len(sequence), len(sequence[0]), len(nucleotide_to_index)))
        for i, seq in enumerate(sequence):
            for j, nucleotide in enumerate(seq):
                if nucleotide == 'N':
                    # In case of wildcards, we assign equal probability to each nucleotide
                    one_hot_sequence[i, j, 0] = 0.25
                    one_hot_sequence[i, j, 1] = 0.25
                    one_hot_sequence[i, j, 2] = 0.25
                    one_hot_sequence[i, j, 3] = 0.25
                elif nucleotide == 'R':
                    one_hot_sequence[i, j, 0] = 0.5
                    one_hot_sequence[i, j, 1] = 0.5
                elif nucleotide == 'Y':
                    one_hot_sequence[i, j, 2] = 0.5
                    one_hot_sequence[i, j, 3] = 0.5
                elif nucleotide == 'K':
                    one_hot_sequence[i, j, 0] = 0.5
                    one_hot_sequence[i, j, 2] = 0.5
                elif nucleotide == 'M':
                    one_hot_sequence[i, j, 1] = 0.5
                    one_hot_sequence[i, j, 3] = 0.5
                elif nucleotide == 'S':
                    one_hot_sequence[i, j, 0] = 0.5
                    one_hot_sequence[i, j, 3] = 0.5
                elif nucleotide == 'W':
                    one_hot_sequence[i, j, 1] = 0.5
                    one_hot_sequence[i, j, 2] = 0.5
                elif nucleotide == 'B':
                    one_hot_sequence[i, j, 0] = 1/3
                    one_hot_sequence[i, j, 2] = 1/3
                    one_hot_sequence[i, j, 3] = 1/3
                elif nucleotide == 'D':
                    one_hot_sequence[i, j, 0] = 1/3
                    one_hot_sequence[i, j, 1] = 1/3
                    one_hot_sequence[i, j, 2] = 1/3
                elif nucleotide == 'H':
                    one_hot_sequence[i, j, 1] = 1/3
                    one_hot_sequence[i, j, 2] = 1/3
                    one_hot_sequence[i, j, 3] = 1/3
                elif nucleotide == 'V':
                    one_hot_sequence[i, j, 0] = 1/3
                    one_hot_sequence[i, j, 1] = 1/3
                    one_hot_sequence[i, j, 3] = 1/3
                else:
                    one_hot_sequence[i, j, nucleotide_to_index[nucleotide]] = 1
        return one_hot_sequence
