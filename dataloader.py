import csv
import os
import pandas as pd
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset

class KgDataset(Dataset):
    def __init__(self, dataset_path):
        self.text_file = f'{dataset_path}/matched_db_numbers.csv'
        self.desc_file = f'{dataset_path}/molecule_descript.csv'

        self.entity_ids = []
        self.smiles = []
        self.sub_smiles = []

        # 读取文本文件并存储数据
        with open(self.text_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    self.entity_ids.append(int(parts[1]))

        with open(self.desc_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 3:
                    self.smiles.append(row[1])
                    self.sub_smiles.append(row[2])
    def __len__(self):
        return len(self.entity_ids)

    def __getitem__(self, idx):
        entity_id = self.entity_ids[idx]
        entity_id = torch.tensor(entity_id)

        smiles = self.smiles[idx]
        sub_smiles = self.sub_smiles[idx]

        # 返回图像路径和对应的图谱中的实体id
        return smiles, sub_smiles, entity_id

class ThreeDDataset(Dataset):
    def __init__(self, dataset_path):
        self.text_file = f'{dataset_path}/3d.csv'
        self.threed_file = f'{dataset_path}/3d_sdf'

        self.data = pd.read_csv(self.text_file)
        assert 'CID' in self.data.columns and 'SMILES' in self.data.columns and 'SUB' in self.data.columns
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['SMILES']
        sub_smiles = row['SUB']
        cid = row['CID']
        sdf_path = os.path.join(self.threed_file, f"{cid}.sdf")
        return smiles, sdf_path, sub_smiles

class ImgDataset(Dataset):
    def __init__(self, dataset_path, preprocess):
        self.text_file = f'{dataset_path}/imag.csv'
        self.img_file = f'{dataset_path}/img'
        self.img_process = preprocess

        self.data = pd.read_csv(self.text_file)
        assert 'CID' in self.data.columns and 'SMILES' in self.data.columns and 'DESC' in self.data.columns and 'SUB' in self.data.columns

    def __len__(self):
        return len(self.data)
        # return 10

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_file, f"{row['CID']}.png")
        image = Image.open(img_path).convert('RGB')
        image = self.img_process(image)
        smiles = row['SMILES']
        sub_smiles = row['SUB']
        # 返回图像路径、文本、SMILES字符串和motif编码序列
        return smiles, image, sub_smiles

class DescDataset(Dataset):
    def __init__(self, dataset_path):
        # 使用pandas读取CSV文件
        self.data = pd.read_csv(f'{dataset_path}/text.csv')
        # 确保所有列都存在
        assert 'SMILES' in self.data.columns and 'Description' in self.data.columns and 'decomposed_smiles' in self.data.columns

    def __len__(self):
        # 返回DataFrame的行数
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引从DataFrame中提取数据
        row = self.data.iloc[idx]
        smiles = row['SMILES']
        desc = row['Description']
        sub_smiles = row['decomposed_smiles']
        return smiles, desc, sub_smiles

class ChemDataset(Dataset):
    def __init__(self, dataset_path):
        self.text_file = f'{dataset_path}/chem.csv'
        self.desc = []
        self.smiles = []
        self.sub_smiles = []

        # 读取文本文件并存储数据
        with open(self.text_file, 'r') as f:
            reader = csv.reader(f)
            for parts in reader:
                if len(parts) == 11:
                    self.smiles.append(parts[1])
                    self.sub_smiles.append(parts[2])
                    self.desc.append(''.join(parts[3:]))

        self.tokens = clip.tokenize(self.desc)

    def __len__(self):
        return len(self.desc)
        # return 10

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        sub_smiles = self.sub_smiles[idx]
        desc = self.tokens[idx]

        return smiles, desc, sub_smiles

