# MKMed

This is the code repository for the paper `Combating the Bucket Effect: Multi-Knowledge Alignment for Medication Recommendation`.

## Datasets

Since the MIMIC-III/IV datasets require authorized access, we are unable to provide the raw data directly here. To obtain these datasets, please complete the official certification process first, then download them from the project's official website.

In addition, for molecular multimodal data, we obtained the SMILES representations of drug molecules from [DrugBank](https://go.drugbank.com/data_packages
)
 and extracted and integrated various types of molecular features from the open-access platform [PubChem](https://pubchem.ncbi.nlm.nih.gov/), ultimately constructing a high-quality dataset tailored for multimodal tasks.

## Documentation

`data`: The data folder stores the dataset and processed files.

* `processing.py`: Processing the Mimic3/4 dataset.
* `mask.py`: Generate `mask.pkl`.

`src`:Storage location of code.
* `main.py`: Main program file, including parameter settings and pre data reading.
* `training.py`: Related training functions.
* `MKMed.py`: Main model file.
* `mole_align.py`: Pre-training model file.
* `gnn` : Support files including GVP and GIN networks.


