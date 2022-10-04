import os
import time

import numpy as np
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys, AllChem


smiles2fp = {}


def get_fp(smiles):
    global smiles2fp
    if not smiles in smiles2fp:
        mol = Chem.MolFromSmiles(smiles)
        macc_fp = MACCSkeys.GenMACCSKeys(mol)  # 167
        rdk_fp = Chem.RDKFingerprint(mol, fpSize=1024)  # 1024
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)  # 1024
        avalon_fp = pyAvalonTools.GetAvalonFP(mol)  # 512
        fp = np.concatenate([macc_fp, rdk_fp, morgan_fp, avalon_fp])
        smiles2fp[smiles] = fp
        # print('Compute fingerprint of', smiles)
    return smiles2fp[smiles]


def to_array(smiles_list):
    return np.concatenate([get_fp(smiles)[None] for smiles in smiles_list], axis=0)


if __name__ == '__main__':
    with open('../../DPI_Data/DA_IC50_ano_specific_model_tsvs/mix_dpi_tsvs/smis.csv', 'r') as reader:
        lines = reader.readlines()
    smiles_list = [line.split(',')[0] for line in lines]
    start = time.clock()
    x = to_array(smiles_list[:1000])
    end = time.clock()
    print('run time', (end - start))
