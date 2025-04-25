import argparse
import os
import clip
import dill
import numpy as np
import torch

from modules.MKMed import MKMed
from training import Test, Train, KgTrain,DescTrain, ImgTrain, ChemFeatureTrain, ThreeDTrain
from utils import buildPrjSmiles,graph_batch_from_smile
from modules.gnn.GNNs import GNNGraph

def set_seed():
    torch.manual_seed(222)
    np.random.seed(222)
    torch.cuda.manual_seed_all(222)
    os.environ['PYTHONHASHSEED'] = str(222)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--debug", default=False)
    parser.add_argument('--pretrain_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrain_saved_path', default="saved_pretrain", type=str)
    # mode
    parser.add_argument("--Test", default=True)
    # environments
    parser.add_argument('--train_epochs', default=25, type=int)
    parser.add_argument('--dataset', default='mimic3')
    parser.add_argument('--train_saved_path', default="saved_train", type=str)
    parser.add_argument('--resume_path',default="")
    # parameters
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument("--regular", type=float, default=0.005, help="regularization parameter")
    parser.add_argument('--target_ddi', type=float, default=0.06, help='expected ddi for training')
    parser.add_argument('--coef', default=2.5, type=float, help='coefficient for DDI Loss Weight Annealing')
    parser.add_argument("--time", default=1)
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    set_seed()
    args = parse_args()

    loss_log_path = f'saved_pretrain/log_{args.time}_{args.device}.txt'

    molecule_path = f'data/{args.dataset}/input/idx2drug.pkl'
    data_path = f'data/{args.dataset}/output/records_final.pkl'
    voc_path = f'data/{args.dataset}/output/voc_final.pkl'
    ddi_adj_path = f'data/{args.dataset}/output/ddi_A_final.pkl'
    mask = f'data/{args.dataset}/output/mask.pkl'
    substruct_smile_path = f'data/{args.dataset}/output/substructure_smiles.pkl'

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(args.device)
    with open(mask, 'rb') as Fin:
        mask = torch.from_numpy(dill.load(Fin)).to(args.device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
        adm_id = 0
        for patient in data:
            for adm in patient:
                adm.append(adm_id)
                adm_id += 1
        if args.debug:
            data = data[:5]
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)
    with open(substruct_smile_path, 'rb') as Fin:
        substruct_smiles_list = dill.load(Fin)

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = [
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    ]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    binary_projection, average_projection, smiles_list = buildPrjSmiles(molecule, med_voc.idx2word)
    voc_size.append(average_projection.shape[1])
    graph = Graph4Visit(data, data_train, voc_size[0], voc_size[1], voc_size[2], args.dataset)

    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(args.device)}

    substruct_graphs = graph_batch_from_smile(substruct_smiles_list)
    substruct_forward = {'batched_data': substruct_graphs.to(args.device)}

    drug_data = {
            'substruct_data': substruct_forward,
            'mol_data': molecule_forward,
            'mask': mask,
            'average_projection': average_projection,
        }

    cross_modal_encoder = GNNGraph(num_layer=3,emb_dim=args.dim,drop_ratio=args.dp,virtual_node=False,device=args.device).to(args.device)

    # clip_net, preprocess = clip.load("ViT-B/32", device=args.device, jit=False)
    # cross_modal_encoder = KgTrain(cross_modal_encoder, clip_net, loss_log_path, args)
    # cross_modal_encoder = ChemFeatureTrain(cross_modal_encoder, clip_net, loss_log_path, args)
    # cross_modal_encoder = ImgTrain(cross_modal_encoder, clip_net, preprocess, loss_log_path, args)
    # cross_modal_encoder = DescTrain(cross_modal_encoder, clip_net, loss_log_path, args)
    # cross_modal_encoder = ThreeDTrain(cross_modal_encoder, clip_net, loss_log_path, args).to(args.device)

    model = MKMed(
        graph=graph,
        cross_modal_encoder=cross_modal_encoder,
        smiles_list=smiles_list,
        tensor_ddi_adj=ddi_adj,
        dropout=args.dp,
        emb_dim=args.dim,
        voc_size=voc_size,
        device=args.device
    ).to(args.device)

    print("1.Training Phase")
    if args.Test:
        print("Test mode, skip training phase")
        with open(args.resume_path, 'rb') as Fin:
            state_dict = torch.load(Fin, map_location=args.device)
        model.load_state_dict(state_dict)

    else:
        model = Train(model, args.device, data_train, data_eval, voc_size, args, drug_data)

    print("2.Testing Phase")
    Test(model, args.device, data_test, voc_size, args, drug_data)
