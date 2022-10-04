import numpy as np
import torch
from torch.utils.data import Dataset


def smi2feats(smi, max_smi_len=102):
    smi = smi.replace(' ', '')
    X = [START_TOKEN]
    for ch in smi[: max_smi_len - 2]:
        X.append(SMI_CHAR_DICT[ch])
    X.append(END_TOKEN)
    X += [PAD_TOKEN] * (max_smi_len - len(X))
    X = np.array(X).astype(np.int64)
    return X

SMI_CHAR_DICT = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,
                ":": 65, "*": 66, "|": 67,
                }
assert np.all(np.array(sorted(list(SMI_CHAR_DICT.values()))) == np.arange(1, len(SMI_CHAR_DICT) + 1))
PAD_TOKEN = 0
START_TOKEN = len(SMI_CHAR_DICT) + 1
END_TOKEN = START_TOKEN + 1
assert PAD_TOKEN not in SMI_CHAR_DICT and START_TOKEN not in SMI_CHAR_DICT.values() and END_TOKEN not in SMI_CHAR_DICT
SMI_CHAR_SET_LEN = len(SMI_CHAR_DICT) + 3  # + (PADDING, START, END)

class MyDataset(Dataset):
    def __init__(self, data_path,):
        with open(data_path, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
        smiles = []
        y_true = []
        for line in lines:
            data = line.strip().split(',')
            smiles.append(data[0])
            y_true.append(float(data[1]))
        self.affinity = y_true


        self.smi = smiles
        assert len(self.affinity) == len(self.smi)
        self.length = len(self.smi)

    def __getitem__(self, idx):
        smiles_feats = self.smi[idx]
        return smiles_feats, torch.Tensor(np.array(self.affinity[idx]))


    def __len__(self):
        return self.length