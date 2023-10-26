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
        # Perform one-hot encoding on DNA sequence
        sequence_one_hot = self.sequence_to_one_hot(sequence)
        return torch.FloatTensor(sequence_one_hot), torch.tensor(class_idx)
    
    def __str__(self):
        return f"BacteriaDataset with {len(self)} items, {len(self.classes)} classes"

    def sequence_to_one_hot(self, sequence):
        # TODO: Consider changing to pure 1,2,3,4 numbers instead of zeros with one 1
        nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'R': 5, 'Y': 6, 'S': 7, 'H': 8, 'W': 9, 'K': 10, 'M': 11, 'B': 12, 'V': 13, 'D': 14}
        one_hot_sequence = np.zeros((len(sequence), len(sequence[0]), len(nucleotide_to_index)))
        print(len(sequence))
        for i, seq in enumerate(sequence):
            for j, nucleotide in enumerate(seq):
                one_hot_sequence[i, j, nucleotide_to_index[nucleotide]] = 1
        return one_hot_sequence
