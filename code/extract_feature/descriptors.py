import random
import time

import torch
from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors
import numpy as np
from functools import lru_cache

def to_array(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    filter_set = set([
        'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_azide', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
        'NumValenceElectrons',
        'fr_diazo', 'fr_isocyan', 'fr_isothiocyan', 'fr_nitroso', 'BertzCT', 'LabuteASA', 'MolMR',
        'fr_prisulfonamd', 'fr_thiocyan', 'fr_aldehyde', 'fr_azo',
        'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_hdrzone',
        'fr_quatN', 'fr_C_S', 'fr_SH', 'fr_alkyl_carbamate', 'fr_barbitur',
        'fr_phos_acid', 'fr_phos_ester', 'fr_term_acetylene', 'fr_Imine',
        'fr_amidine', 'fr_guanido', 'fr_hdrzine', 'fr_imide', 'fr_lactam',
        'fr_lactone', 'fr_oxime', 'EState_VSA11', 'fr_N_O', 'fr_nitro',
        'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'MaxPartialCharge',
        'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'Ipc',
        'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI',
        'BCUT2D_MRLOW'
    ])
    features = {}
    for smiles, mol in zip(smiles_list, mols):
        feature = []
        for desc in Descriptors._descList:
            if desc[0] in filter_set:
                continue
            print(desc[0])
            feature.append(desc[1](mol))
            if np.isnan(feature[-1]):
                print(desc)
                raise Exception('233')
        features[smiles] = np.array(feature)
    # np.save('../gnn_dataset/descriptors.npy', features)
    return features


@lru_cache()
def to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    filter_set = set([
        'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_azide','MolWt','HeavyAtomMolWt','ExactMolWt','NumValenceElectrons',
        'fr_diazo', 'fr_isocyan', 'fr_isothiocyan', 'fr_nitroso','BertzCT','LabuteASA','MolMR',
        'fr_prisulfonamd', 'fr_thiocyan', 'fr_aldehyde', 'fr_azo',
        'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_hdrzone',
        'fr_quatN', 'fr_C_S', 'fr_SH', 'fr_alkyl_carbamate', 'fr_barbitur',
        'fr_phos_acid', 'fr_phos_ester', 'fr_term_acetylene', 'fr_Imine',
        'fr_amidine', 'fr_guanido', 'fr_hdrzine', 'fr_imide', 'fr_lactam',
        'fr_lactone', 'fr_oxime', 'EState_VSA11', 'fr_N_O', 'fr_nitro',
        'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'MaxPartialCharge',
        'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge','Ipc',
        'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW'
    ])
    feature = []
    for desc in Descriptors._descList:
        if desc[0] in filter_set:
            continue
        feature.append(desc[1](mol))
        if desc[1](mol)>100:
            print(len(feature), desc[0])
        if np.isnan(feature[-1]):
            print(desc)
            raise Exception('233')
    feature = np.array(feature)
    # np.save('../gnn_dataset/descriptors.npy', feature)
    return feature

if __name__ == '__main__':
    with open('../../DPI_Data/DA_IC50_ano_specific_model_tsvs/mix_dpi_tsvs/smis.csv', 'r') as reader:
        lines = reader.readlines()
    desc = torch.load('../gnn_dataset/descriptors.pt')
    print()
    smiles_list = [line.split(',')[0] for line in lines]
    sam = random.sample(smiles_list, 1)
    # start = time.clock()
    x = to_array(sam)
    # end = time.clock()
    # start2 = time.clock()
    x2 = [desc[smi] for smi in smiles_list]
    # end2 = time.clock()
    # print('run time', (end - start))
    # print('run time', (end2 - start2))
    #
    # smi = 'CC(=O)NC1C(=O)NC2=CC=CC=C2C(=N1)C3=CC=CC=C3F'
    # mol = Chem.MolFromSmiles(smi)
    # molwt = Descriptors.MolWt(mol)
    # print(molwt)

