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
# tempF = set()
# m = BRICS.BRICSDecompose(Chem.MolFromSmiles('[H][C@@]12C[C@@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'))
# for frac in m:
#     tempF.add(frac)
# print(tempF)