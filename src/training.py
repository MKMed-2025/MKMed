import dill
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.dataloader import KgDataset, DescDataset, ChemDataset, ThreeDDataset, ImgDataset
from modules.mole_align import KG_align, Img_align, Desc_align, Chem_align, ThreeD_align
from utils import llprint, multi_label_metric, ddi_rate_score, get_n_params, parameter_report, Regularization

def KgTrain(cross_modal_encoder, clip_net, loss_log_path, args):
    if args.debug:
        dataset_path = 'data/kg_data/dataset_100'
    else:
        dataset_path = 'data/kg_data/dataset_all'

    kg_dataset = KgDataset(dataset_path=dataset_path)
    kg_dataloader = DataLoader(kg_dataset, batch_size=args.batch_size, shuffle=True)

    model = KG_align(cross_modal_encoder, clip_net, args.dim, args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_func = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model = model.state_dict()
    with open(loss_log_path, 'a') as f:
        f.write(f'KG Aligning\n')

    EPOCH = args.pretrain_epochs
    if args.debug:
        EPOCH = 3
    for epoch in range(EPOCH):

        total_loss = 0

        with torch.cuda.amp.autocast(enabled=True), tqdm(kg_dataloader, desc=f'Epoch {epoch + 1}/{EPOCH}',
                                                         unit='batch') as tepoch:
            for smiles, sub_smiles ,entity_id in tepoch:
                optimizer.zero_grad()

                logits = model(smiles, sub_smiles, entity_id)

                ground_truth = torch.arange(len(smiles), dtype=torch.long, device=args.device)
                loss = (loss_func(logits, ground_truth) + loss_func(logits.t(), ground_truth)) / 2

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = total_loss / len(kg_dataloader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            logger.info(f"Epoch {epoch + 1}: The best model has been saved, and the loss value is: {best_loss}")

        with open(loss_log_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}: Loss: {epoch_loss:.4f}\n')

        logger.info(f'Epoch {epoch + 1} Loss: {epoch_loss:.4f}')

    torch.save(best_model, f'{args.pretrain_saved_path}/kg_model_{args.time}_{args.device}.pth')
    return model.cross_modal_encoder

def ImgTrain(cross_modal_encoder, clip_net, preprocess, loss_log_path, args):
    if args.debug:
        dataset_path = 'data/image_data/dataset_100'
    else:
        dataset_path = 'data/image_data/dataset_all'

    img_dataset = ImgDataset(dataset_path=dataset_path, preprocess=preprocess)
    img_dataloader = DataLoader(img_dataset, batch_size=args.batch_size, shuffle=True)

    model = Img_align(cross_modal_encoder, clip_net, args.dim, args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_func = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model = model.state_dict()
    with open(loss_log_path, 'a') as f:
        f.write(f'Image Aligning\n')

    EPOCH = args.pretrain_epochs
    if args.debug:
        EPOCH = 3
    for epoch in range(EPOCH):

        total_loss = 0

        with torch.cuda.amp.autocast(enabled=True), \
                tqdm(img_dataloader, desc=f'Epoch {epoch + 1}/{EPOCH}', unit='batch') as tepoch:

            for smiles, img, sub_smiles in tepoch:
                optimizer.zero_grad()

                logits = model(smiles, img, sub_smiles)

                ground_truth = torch.arange(len(img), dtype=torch.long, device=args.device)
                loss = (loss_func(logits, ground_truth) + loss_func(logits.t(), ground_truth)) / 2

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = total_loss / len(img_dataloader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            logger.info(f"Epoch {epoch + 1}: The best model has been saved, and the loss value is: {best_loss}")

        with open(loss_log_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}: Loss: {epoch_loss:.4f}\n')

        logger.info(f'Epoch {epoch + 1} Loss: {epoch_loss:.4f}')

    torch.save(best_model, f'{args.pretrain_saved_path}/img_model_{args.time}_{args.device}.pth')
    return model.cross_modal_encoder

def DescTrain(cross_modal_encoder, clip_net, loss_log_path, args):
    if args.debug:
        dataset_path = 'data/desc_data/dataset_100'
    else:
        dataset_path = 'data/desc_data/dataset_all'

    desc_dataset = DescDataset(dataset_path=dataset_path)
    desc_dataloader = DataLoader(desc_dataset, batch_size=args.batch_size, shuffle=True)

    model = Desc_align(cross_modal_encoder, clip_net, args.dim, args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_func = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model = model.state_dict()
    with open(loss_log_path, 'a') as f:
        f.write(f'Desc Aligning\n')

    EPOCH = args.pretrain_epochs
    if args.debug:
        EPOCH = 3
    for epoch in range(EPOCH):

        total_loss = 0

        with torch.cuda.amp.autocast(enabled=True), \
                tqdm(desc_dataloader, desc=f'Epoch {epoch + 1}/{EPOCH}', unit='batch') as tepoch:
            for smiles, desc, sub_smiles in tepoch:
                optimizer.zero_grad()

                logits = model(smiles, desc, sub_smiles)

                ground_truth = torch.arange(len(smiles), dtype=torch.long, device=args.device)
                loss = (loss_func(logits, ground_truth) + loss_func(logits.t(), ground_truth)) / 2

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = total_loss / len(desc_dataloader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            logger.info(f"Epoch {epoch + 1}: The best model has been saved, and the loss value is: {best_loss}")

        with open(loss_log_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}: Loss: {epoch_loss:.4f}\n')

        logger.info(f'Epoch {epoch + 1} Loss: {epoch_loss:.4f}')

    torch.save(best_model, f'{args.pretrain_saved_path}/desc_model_{args.time}_{args.device}.pth')
    return model.cross_modal_encoder

def ChemFeatureTrain(cross_modal_encoder, clip_net, loss_log_path, args):
    if args.debug:
        dataset_path = 'data/motif_data/dataset_100'
    else:
        dataset_path = 'data/motif_data/dataset_all'

    chem_dataset = ChemDataset(dataset_path=dataset_path)
    chem_dataloader = DataLoader(chem_dataset, batch_size=args.batch_size, shuffle=True)

    model = Chem_align(cross_modal_encoder, clip_net, args.dim, args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_func = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model = model.state_dict()
    with open(loss_log_path, 'a') as f:
        f.write(f'Chem Aligning\n')

    EPOCH = args.pretrain_epochs
    if args.debug:
        EPOCH = 3
    for epoch in range(EPOCH):

        total_loss = 0

        with torch.cuda.amp.autocast(enabled=True), \
                tqdm(chem_dataloader, desc=f'Epoch {epoch + 1}/{EPOCH}', unit='batch') as tepoch:
            for smiles, desc, sub_smiles in tepoch:
                optimizer.zero_grad()

                logits = model(smiles, desc, sub_smiles)

                ground_truth = torch.arange(len(smiles), dtype=torch.long, device=args.device)
                loss = (loss_func(logits, ground_truth) + loss_func(logits.t(), ground_truth)) / 2

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = total_loss / len(chem_dataloader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            logger.info(f"Epoch {epoch + 1}: The best model has been saved, and the loss value is: {best_loss}")

        # 将损失值写入日志文件
        with open(loss_log_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}: Loss: {epoch_loss:.4f}\n')

        logger.info(f'Epoch {epoch + 1} Loss: {epoch_loss:.4f}')

    torch.save(best_model, f'{args.pretrain_saved_path}/chem_model_{args.time}_{args.device}.pth')
    return model.cross_modal_encoder

def ThreeDTrain(cross_modal_encoder, clip_net, loss_log_path, args):
    if args.debug:
        dataset_path = 'data/3d_data/dataset_100'
    else:
        dataset_path = 'data/3d_data/dataset_all'
    ThreeD_dataset = ThreeDDataset(dataset_path=dataset_path)
    ThreeD_data = DataLoader(ThreeD_dataset, batch_size=args.batch_size, shuffle=False)

    model = ThreeD_align(cross_modal_encoder, clip_net,args.dim, args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_func = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model = model.state_dict()
    with open(loss_log_path, 'a') as f:
        f.write(f'3D Aligning\n')

    EPOCH = args.pretrain_epochs
    if args.debug:
        EPOCH = 3
    for epoch in range(EPOCH):

        total_loss = 0

        with torch.cuda.amp.autocast(enabled=True), \
                tqdm(ThreeD_data, desc=f'Epoch {epoch + 1}/{EPOCH}', unit='batch') as tepoch:
            for smiles,threed,sub_smiles in tepoch:
                optimizer.zero_grad()

                logits = model(smiles,threed,sub_smiles)

                ground_truth = torch.arange(len(smiles), dtype=torch.long, device=args.device)
                loss = (loss_func(logits, ground_truth) + loss_func(logits.t(), ground_truth)) / 200

                total_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = total_loss / len(smiles)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            logger.info(f"Epoch {epoch + 1}: The best model has been saved, and the loss value is: {best_loss}")

        with open(loss_log_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}: Loss: {epoch_loss:.4f}\n')

        logger.info(f'Epoch {epoch + 1} Loss: {epoch_loss:.4f}')

    torch.save(best_model, f'{args.pretrain_saved_path}/3d_model_{args.time}_{args.device}.pth')
    return model.cross_modal_encoder


def eval_one_epoch(model, data_eval, voc_size, args, drug_data):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0
    for step, input_seq in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_seq):
            output, _ = model(input_seq[:adm_idx + 1],
                              **drug_data)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output = torch.sigmoid(output).detach().cpu().numpy()[0]
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step:: {} / {}'.format(step + 1, len(data_eval)))

    ddi_rate = ddi_rate_score(smm_record, args)
    output_str = '\nDDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' + \
                 'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'
    llprint(output_str.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


def Test(model, device, data_test, voc_size, args, drug_data):
    model = model.to(device).eval()
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    np.random.seed(0)
    for _ in range(10):
        selected_indices = np.random.choice(len(data_test), size=round(len(data_test) * 0.8), replace=True)
        selected_indices_list = selected_indices.tolist()
        test_sample = [data_test[i] for i in selected_indices_list]
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval_one_epoch(model, test_sample, voc_size, args, drug_data)
        result.append([ja, ddi_rate, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ja', 'ddi_rate', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])

    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))


def Train(model, device, data_train, data_eval, voc_size, args, drug_data):
    regular = Regularization(model, args.regular, p=0)  # 正则化模型

    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best = {"epoch": 0, "ja": 0, "ddi": 0, "prauc": 0, "f1": 0, "med": 0, 'model': model}
    total_train_time, ddi_losses, ddi_values = 0, [], []

    EPOCH = args.train_epochs
    if args.debug:
        EPOCH = 3
    for epoch in range(EPOCH):
        print(f'----------------Epoch {epoch + 1}------------------')
        model = model.train()
        tic, ddi_losses_epoch = time.time(), []
        for step, input_seq in enumerate(data_train):
            for adm_idx, adm in enumerate(input_seq):
                bce_target = torch.zeros((1, voc_size[2])).to(device)
                bce_target[:, adm[2]] = 1

                multi_target = -torch.ones((1, voc_size[2])).long()
                for idx, item in enumerate(adm[2]):
                    multi_target[0][idx] = item
                multi_target = multi_target.to(device)

                result, loss_ddi = model(patient_data = input_seq[:adm_idx + 1],
                                         **drug_data)

                sigmoid_res = torch.sigmoid(result)

                loss_bce = binary_cross_entropy_with_logits(result, bce_target)
                loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)

                result = sigmoid_res.detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]], args)

                if current_ddi_rate <= args.target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = args.coef * (1 - (current_ddi_rate / args.target_ddi))
                    beta = min(math.exp(beta), 1)
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        ddi_losses.append(sum(ddi_losses_epoch) / len(ddi_losses_epoch))
        print(f'\nddi_loss : {ddi_losses[-1]}\n')
        train_time, tic = time.time() - tic, time.time()
        total_train_time += train_time
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval_one_epoch(model, data_eval, voc_size, args, drug_data)
        print(f'training time: {train_time}, testing time: {time.time() - tic}')
        ddi_values.append(ddi_rate)
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
            ))

        if epoch != 0:
            if best['ja'] < ja:
                best['epoch'] = epoch
                best['ja'] = ja
                best['model'] = model
                best['ddi'] = ddi_rate
                best['prauc'] = prauc
                best['f1'] = avg_f1
                best['med'] = avg_med
            print("best_epoch: {}, best_ja: {:.4f}".format(best['epoch'], best['ja']))
        # graph_report(history, args)

    print('avg training time/epoch: {:.4f}'.format(total_train_time / EPOCH))
    parameter_report(best, regular, args)

    torch.save(best['model'].state_dict(),
               "{}/{}/trained_model_{}_{}_{:.4f}".format(args.train_saved_path, args.dataset,args.time,args.device, best['ja']))
    return best['model']
