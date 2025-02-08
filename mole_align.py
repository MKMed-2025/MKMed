import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch_geometric.data import Data, Batch
from utils import gen_sub_projection, graph_batch_from_smile, drug_sdf_db
from gnn.GNNs import DrugGVPModel
from SetTransformer import SAB

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = Attn.to(torch.float32)
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        O = torch.matmul(Attn, other_feat)

        return O
class KG_align(nn.Module):
    def __init__(self, mole_encoder, clip_net, emb_dim,device):
        super(KG_align, self).__init__()
        self.device = device
        self.emb_dim = emb_dim

        pretrained_embedding = torch.from_numpy(np.load("data/kg_data/dataset_all/DRKG_TransE_l2_entity.npy")).to(
            self.device)
        num_entities, embedding_dim = pretrained_embedding.size()
        self.kg_embedding = nn.Embedding(num_entities, embedding_dim, padding_idx=None)
        self.kg_embedding.weight = nn.Parameter(pretrained_embedding)

        self.kg_embedding.weight.requires_grad = False

        self.mole_encoder = mole_encoder
        self.clip_net = clip_net

        self.sab = SAB(self.emb_dim, self.emb_dim, 2, use_ln=True).to(self.device)
        self.aggregator = AdjAttenAgger(
            self.emb_dim, self.emb_dim, max(self.emb_dim, self.emb_dim)
        ).to(self.device)

        self.entity_linear = nn.Linear(400, self.emb_dim).to(self.device)
        self.graph_linear = nn.Linear(self.emb_dim, self.emb_dim).to(self.device)

    def forward(self, smiles, sub_smiles, entity_id):
        smiles_list = list(smiles)
        sub_list = list(sub_smiles)
        entity_id = entity_id.to(self.device)

        sub_list, sub_projection_matrix = gen_sub_projection(smiles_list, sub_list)
        sub_projection_matrix = torch.from_numpy(sub_projection_matrix).to(self.device)

        smiles_graph = graph_batch_from_smile(smiles_list).to(self.device)
        sub_graph = graph_batch_from_smile(sub_list).to(self.device)

        mole_features = self.mole_encoder(smiles_graph)  # [32,64]
        sub_features = self.sab(self.mole_encoder(sub_graph).unsqueeze(0)).squeeze(0)  # [,64]
        smiles_features = self.aggregator(mole_features, sub_features,
                                          mask=torch.logical_not(sub_projection_matrix > 0))

        entity_features = self.kg_embedding(entity_id)

        smiles_features = self.graph_linear(smiles_features)
        entity_features = self.entity_linear(entity_features)

        # normalized features
        smiles_features = smiles_features / smiles_features.norm(dim=1, keepdim=True)
        entity_features = entity_features / entity_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_net.logit_scale.exp()
        logit_img2desc = logit_scale * smiles_features @ entity_features.t()

        return logit_img2desc

class Img_align(nn.Module):
    def __init__(self, mole_encoder, clip_net,emb_dim, device):
        super(Img_align, self).__init__()
        self.device = device
        self.mole_encoder = mole_encoder
        self.clip_net = clip_net
        self.sab = SAB(emb_dim, emb_dim, 2, use_ln=True).to(self.device)
        self.aggregator = AdjAttenAgger(
            emb_dim, emb_dim, max(emb_dim, emb_dim)
        ).to(self.device)

        self.graph_linear = nn.Linear(emb_dim, 64).to(self.device)
        self.img_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        ).to(self.device)

    def forward(self, smiles_list, image, sub_list):
        smiles_list = list(smiles_list)
        sub_list = list(sub_list)
        image = image.to(self.device)

        sub_list, sub_projection_matrix = gen_sub_projection(smiles_list, sub_list)
        sub_projection_matrix = torch.from_numpy(sub_projection_matrix).to(self.device)

        smiles_graph = graph_batch_from_smile(smiles_list).to(self.device)
        sub_graph = graph_batch_from_smile(sub_list).to(self.device)

        mole_features = self.mole_encoder(smiles_graph)  # [32,64]
        sub_features = self.sab(self.mole_encoder(sub_graph).unsqueeze(0)).squeeze(0)  # [,64]
        smiles_features = self.aggregator(mole_features, sub_features, mask=torch.logical_not(sub_projection_matrix > 0))
        image_features = self.clip_net.encode_image(image)

        image_features = self.img_linear(image_features)
        smiles_features = self.graph_linear(smiles_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        smiles_features = smiles_features / smiles_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_net.logit_scale.exp()

        logit = logit_scale * image_features @ smiles_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logit

class Desc_align(nn.Module):
    def __init__(self, mole_encoder, clip_net, emb_dim,device):
        super(Desc_align, self).__init__()
        self.device = device
        self.mole_encoder = mole_encoder
        self.clip_net = clip_net

        self.sab = SAB(emb_dim, emb_dim, 2, use_ln=True).to(self.device)
        self.aggregator = AdjAttenAgger(
            emb_dim, emb_dim, max(emb_dim, emb_dim)
        ).to(self.device)
        self.graph_linear = nn.Linear(emb_dim, 64).to(self.device)
        self.desc_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        ).to(self.device)

    def process_long_texts(self, texts, max_sentences_per_part=3):
        features_list = []

        for text in texts:

            sentences = text.split('.')
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            parts = ['.'.join(sentences[i:i + max_sentences_per_part]) for i in
                     range(0, len(sentences), max_sentences_per_part)]


            part_features_list = []
            for part in parts:
                if len(part) > 0:
                    try:
                        tokenized_part = clip.tokenize([part]).to(self.device)  # 标记化
                    except RuntimeError as e:
                        if "context length 77" in str(e):

                            tokenized_part = clip.tokenize([part[:77]]).to(self.device)
                        else:

                            raise e

                    with torch.no_grad():
                        features = self.clip_net.encode_text(tokenized_part)
                    part_features_list.append(features)

            if part_features_list:
                all_part_features = torch.cat(part_features_list, dim=0)
                average_features = all_part_features.mean(dim=0)
                features_list.append(average_features)

        return torch.stack(features_list)

    def forward(self, smiles_list, desc, sub_list):
        smiles_list = list(smiles_list)
        sub_list = list(sub_list)
        sub_list, sub_projection_matrix = gen_sub_projection(smiles_list,sub_list)
        sub_projection_matrix = torch.from_numpy(sub_projection_matrix).to(self.device)

        smiles_graph = graph_batch_from_smile(smiles_list).to(self.device)
        sub_graph = graph_batch_from_smile(sub_list).to(self.device)

        mole_features = self.mole_encoder(smiles_graph) # [32,64]
        sub_features = self.sab(self.mole_encoder(sub_graph).unsqueeze(0)).squeeze(0)
        mole_features = self.aggregator(mole_features, sub_features, mask=torch.logical_not(sub_projection_matrix > 0))

        desc_features = self.process_long_texts(desc,1)

        mole_features = self.graph_linear(mole_features)
        desc_features = self.desc_linear(desc_features)
        # normalized features
        mole_features = mole_features / mole_features.norm(dim=1, keepdim=True)
        desc_features = desc_features / desc_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_net.logit_scale.exp()
        logits = logit_scale * mole_features @ desc_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits

class Chem_align(nn.Module):
    def __init__(self, mole_encoder, clip_net,emb_dim, device):
        super(Chem_align, self).__init__()
        self.device = device
        self.mole_encoder = mole_encoder
        self.clip_net = clip_net
        self.sab = SAB(emb_dim, emb_dim, 2, use_ln=True).to(self.device)
        self.aggregator = AdjAttenAgger(
            emb_dim, emb_dim, max(emb_dim, emb_dim)
        ).to(self.device)

        self.desc_encoder = nn.Sequential(
            nn.Linear(77, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        ).to(self.device)

        self.graph_linear = nn.Linear(emb_dim, 64).to(self.device)

        self.desc_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        ).to(self.device)

    def forward(self, smiles_list, desc, sub_list):
        smiles_list = list(smiles_list)
        sub_list = list(sub_list)
        desc = desc.float().to(self.device)

        sub_list, sub_projection_matrix = gen_sub_projection(smiles_list, sub_list)
        sub_projection_matrix = torch.from_numpy(sub_projection_matrix).to(self.device)

        smiles_graph = graph_batch_from_smile(smiles_list).to(self.device)
        sub_graph = graph_batch_from_smile(sub_list).to(self.device)

        mole_features = self.mole_encoder(smiles_graph)  # [32,64]
        sub_features = self.sab(self.mole_encoder(sub_graph).unsqueeze(0)).squeeze(0)  # [,64]
        mole_features = self.aggregator(mole_features, sub_features, mask=torch.logical_not(sub_projection_matrix > 0))
        desc_features = self.desc_encoder(desc)

        mole_features = self.graph_linear(mole_features)
        desc_features = self.desc_linear(desc_features)
        # normalized features
        mole_features = mole_features / mole_features.norm(dim=1, keepdim=True)
        desc_features = desc_features / desc_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_net.logit_scale.exp()
        logits = logit_scale * mole_features @ desc_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits

class ThreeD_align(torch.nn.Module):
    def __init__(self, mol_encoder, clip_net, emb_dim,device=torch.device('cuda:0')):
        super(ThreeD_align, self).__init__()
        self.device = device
        self.sab = SAB(emb_dim, emb_dim, 2, use_ln=True).to(self.device)
        self.aggregator = AdjAttenAgger(
            emb_dim, emb_dim, max(emb_dim, emb_dim)
        ).to(self.device)
        self.drug_node_in_dim = [66, 1]
        self.drug_node_h_dims = [emb_dim, 64]
        self.drug_edge_in_dim = [16, 1]
        self.drug_edge_h_dims = [32, 1]
        self.drug_fc_dims = [1024, emb_dim]
        self.drug_emb_dim = self.drug_node_h_dims[0]

        self.mole_encoder = mol_encoder
        self.clip_net = clip_net
        self.drug_model = DrugGVPModel(
            node_in_dim=self.drug_node_in_dim, node_h_dim=self.drug_node_h_dims,
            edge_in_dim=self.drug_edge_in_dim, edge_h_dim=self.drug_edge_h_dims,
            device=self.device
        ).to(self.device)

        self.drug_fc = self.get_fc_layers(
            [self.drug_emb_dim] + self.drug_fc_dims,
            dropout=0.25, batchnorm=False,
            no_last_dropout=True, no_last_activation=True).to(self.device)


    def get_fc_layers(self, hidden_sizes,
                      dropout=0, batchnorm=False,
                      no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(torch.nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(torch.nn.BatchNorm1d(out_dim))
        return torch.nn.Sequential(*layers)

    def forward(self, mol_data, drug_geo_list,sub_list):
        sub_list = list(sub_list)
        sub_list, sub_projection_matrix = gen_sub_projection(mol_data, sub_list)
        sub_projection_matrix = torch.from_numpy(sub_projection_matrix).to(self.device)

        smiles = graph_batch_from_smile(mol_data).to(self.device)
        sub_graph = graph_batch_from_smile(sub_list).to(self.device)


        global_embeddings = self.mole_encoder(smiles)
        sub_features = self.sab(self.mole_encoder(sub_graph).unsqueeze(0)).squeeze(0)  # [,64]
        global_embeddings = self.aggregator(global_embeddings, sub_features, mask=torch.logical_not(sub_projection_matrix > 0))

        drug_geo_dict = drug_sdf_db(drug_geo_list)
        drug_geo_list = [x.to(self.device) for x in drug_geo_dict.values()]
        drug_batch = Batch.from_data_list(drug_geo_list)
        xd = self.drug_model(drug_batch)

        xd = self.drug_fc(xd)  # [284,64]

        global_embeddings = F.normalize(global_embeddings, p=2, dim=1)
        xd = F.normalize(xd, p=2, dim=1)

        # cosine similarity as logits
        logit_scale = self.clip_net.logit_scale.exp()
        logits = logit_scale * global_embeddings @ xd.t()

        return logits


