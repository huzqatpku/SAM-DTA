import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from extract_feature import fingerprint, descriptors

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


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor):

        return input.squeeze(self.dim)


class CDilated(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output


class DilatedParllelResidualBlockB(nn.Module):

    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        # merge
        combine = torch.cat([d1, add1, add2, add3], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class SMILES_CONV(nn.Module):

    def __init__(
            self, num_tasks=1,
            hidden=128, num_layers=2,
            fc_hidden=-1, use_fp=False, use_desc=False, **kwargs
    ):
        super().__init__()
        # layers for smiles
        assert use_fp in [True, False]
        self.use_desc = use_desc
        self.use_fp = use_fp
        if fc_hidden == -1:
            fc_hidden = 128
        self.fc_hidden = fc_hidden
        self.embed_smi = nn.Embedding(70, hidden//2)
        conv_smi = []


        conv_smi.append(DilatedParllelResidualBlockB(hidden//2, hidden))
        for i in range(num_layers - 1):
            conv_smi.append(DilatedParllelResidualBlockB(hidden, hidden))
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze(-1))
        self.conv_smi = nn.Sequential(*conv_smi)
        self.num_tasks = num_tasks
        #
        lin1_input = hidden
        if self.use_fp and self.use_desc:
            lin1_input = hidden + 2727 + 151
        elif not self.use_fp and self.use_desc:
            lin1_input = hidden + 151
        elif self.use_fp and not self.use_desc:
            lin1_input = hidden + 2727


        self.lin1 = nn.Sequential(
            nn.Linear(lin1_input, fc_hidden),
            # nn.Dropout(0.5),
            nn.Dropout(0.1),
            nn.PReLU(),
        )
        self.lin2 = nn.Linear(fc_hidden, num_tasks)





    def forward(self, smiles_list, descriptors=None, extract_feature=False):
        smi_tensors = torch.tensor(np.array([smi2feats(smi) for smi in smiles_list])).long().cuda()
        # print('smi_tensors', smi_tensors.shape)
        smi_embed = self.embed_smi(smi_tensors)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)
        if self.use_fp and self.use_desc:
            features = fingerprint.to_array(smiles_list)
            features = torch.tensor(features).float().cuda()
            descriptors = descriptors.float()
            smi_conv = torch.cat([smi_conv, features, descriptors], dim=1)
        elif not self.use_fp and self.use_desc:
            descriptors = descriptors.float()
            smi_conv = torch.cat([smi_conv, descriptors], dim=1)
        elif self.use_fp and not self.use_desc:
            features = fingerprint.to_array(smiles_list)
            features = torch.tensor(features).float().cuda()
            smi_conv = torch.cat([smi_conv, features], dim=1)

        x = self.lin1(smi_conv)
        if extract_feature:
            return x
        x = self.lin2(x)
        x = [x[:, task_ind, None] for task_ind in range(self.num_tasks)]
        return x






