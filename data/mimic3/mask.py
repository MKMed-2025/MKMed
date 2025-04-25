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

matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))

for i, fracList in enumerate(fraction):
    for frac in fracList:
        matrix[i, fracSet.index(frac)] = 1

dill.dump(matrix, open('output/mask.pkl', 'wb'))
dill.dump(fracSet, open('output/substructure_smiles.pkl', 'wb'))
