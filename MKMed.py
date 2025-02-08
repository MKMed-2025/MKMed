import torch
import torch.nn as nn
import math
from SetTransformer import SAB

class Review(nn.Module):
    def __init__(self, graph, num_med):
        super(Review, self).__init__()

        self.num_med = num_med
        self.c1 = graph
        diag_med_high = graph.get_threshold_effect(0.97, "Diag", "Med")
        diag_med_low = graph.get_threshold_effect(0.90, "Diag", "Med")
        proc_med_high = graph.get_threshold_effect(0.97, "Proc", "Med")
        proc_med_low = graph.get_threshold_effect(0.90, "Proc", "Med")
        self.c1_high_limit = nn.Parameter(torch.tensor([diag_med_high, proc_med_high]))
        self.c1_low_limit = nn.Parameter(torch.tensor([diag_med_low, proc_med_low]))
        self.c1_minus_weight = nn.Parameter(torch.tensor(0.01))
        self.c1_plus_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, pre_prob, diags, procs):
        reviewed_prob = pre_prob.clone()

        for m in range(self.num_med):
            max_cdm = 0.0
            max_cpm = 0.0
            for d in diags:
                cdm = self.c1.get_effect(d, m, "Diag", "Med")
                max_cdm = max(max_cdm, cdm)
            for p in procs:
                cpm = self.c1.get_effect(p, m, "Proc", "Med")
                max_cpm = max(max_cpm, cpm)

            if max_cdm < self.c1_low_limit[0] and max_cpm < self.c1_low_limit[1]:
                reviewed_prob[0, m] -= self.c1_minus_weight
            elif max_cdm > self.c1_high_limit[0] or max_cpm > self.c1_high_limit[1]:
                reviewed_prob[0, m] += self.c1_plus_weight

        return reviewed_prob

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, mask=None):
        Q = self.Qdense(main_feat)      # 131*64
        K = self.Kdense(other_feat)     # 492*64
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        O = torch.matmul(Attn, other_feat)

        return O

class MKMed(torch.nn.Module):
    def __init__(
            self,
            graph,
            mole_encoder,
            smiles_list,
            tensor_ddi_adj,
            emb_dim,
            voc_size,
            dropout,
            device=torch.device('cpu'),
    ):
        super(MKMed, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.smiles_list = smiles_list

        # Embedding of all entities
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim),
            torch.nn.Embedding(voc_size[3], emb_dim)
        ])

        self.mole_encoder = mole_encoder
        self.sub_encoder = mole_encoder

        self.sab = SAB(emb_dim, emb_dim, 2, use_ln=True)
        self.aggregator = AdjAttenAgger(
            emb_dim, emb_dim, max(emb_dim, emb_dim)
        )

        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()

        self.graph = graph

        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])

        self.review = Review(self.graph, voc_size[0], voc_size[1], voc_size[2])

        # Convert patient information to drug score
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 6, voc_size[2])
        )

        self.linear = nn.Linear(128,64)

        self.tensor_ddi_adj = tensor_ddi_adj
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def forward(self, patient_data, mol_data, average_projection, substruct_data, ddi_mask_H):
        seq_diag, seq_proc, seq_med = [], [], []
        for adm_id, adm in enumerate(patient_data):


            global_mol = self.mole_encoder(**mol_data)
            global_mol = torch.mm(torch.FloatTensor(average_projection).to(self.device), global_mol)
            sub_mol = self.sab(self.sub_encoder(**substruct_data).unsqueeze(0)).squeeze(0)
            emb_mole = self.aggregator(global_mol, sub_mol,mask=torch.logical_not(ddi_mask_H > 0))

            idx_diag = torch.LongTensor(adm[0]).to(self.device)
            idx_proc = torch.LongTensor(adm[1]).to(self.device)
            emb_diag = self.rnn_dropout(self.embeddings[0](idx_diag)).unsqueeze(0)
            emb_proc = self.rnn_dropout(self.embeddings[1](idx_proc)).unsqueeze(0)

            if adm == patient_data[0]:
                emb_med2 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient_data[adm_id - 1]
                emb_med2 = self.rnn_dropout(emb_mole[adm_last[2],:]).unsqueeze(0)

            seq_diag.append(torch.sum(emb_diag, keepdim=True, dim=1))
            seq_proc.append(torch.sum(emb_proc, keepdim=True, dim=1))
            seq_med.append(torch.sum(emb_med2, keepdim=True, dim=1))

        seq_diag = torch.cat(seq_diag, dim=1)
        seq_proc = torch.cat(seq_proc, dim=1)
        seq_med = torch.cat(seq_med, dim=1)
        output_diag, hidden_diag = self.seq_encoders[0](seq_diag)
        output_proc, hidden_proc = self.seq_encoders[1](seq_proc)
        output_med, hidden_med = self.seq_encoders[2](seq_med)
        seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med], dim=-1)
        last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        score = self.query(patient_repr).unsqueeze(0)
        score = self.review(score, patient_data[-1][0], patient_data[-1][1])

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return score, batch_neg
