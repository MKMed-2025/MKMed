from rdkit import Chem
from rdkit.Chem import BRICS
import dill
import numpy as np

NDCList = dill.load(open('input/idx2drug.pkl', 'rb'))
voc = dill.load(open('output/voc_final.pkl', 'rb'))
med_voc = voc['med_voc']

fraction = []
for k, v in med_voc.idx2word.items():
    tempF = set()

    for SMILES in NDCList[v]:
        try:
            m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
            for frac in m:
                tempF.add(frac)
        except:
            pass

    fraction.append(tempF)

fracSet = []
for i in fraction:
    fracSet += i
fracSet = list(set(fracSet))

ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))

for i, fracList in enumerate(fraction):
    for frac in fracList:
        ddi_matrix[i, fracSet.index(frac)] = 1

# row_sums = ddi_matrix.sum(axis=1)
# average_ddi_matrix = np.zeros_like(ddi_matrix)
# for i in range(ddi_matrix.shape[0]):
#     if row_sums[i] != 0:
#         inverse_sum = 1 / row_sums[i]
#         average_ddi_matrix[i] = np.where(ddi_matrix[i]!=0, inverse_sum, 0)
#
# dill.dump(average_ddi_matrix,open('output/average_ddi_matrix.pkl', 'wb'))
dill.dump(ddi_matrix, open('output/ddi_mask_H.pkl', 'wb'))
dill.dump(fracSet, open('output/substructure_smiles.pkl', 'wb'))
