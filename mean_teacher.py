import math
import warnings

import os
import numpy as np
import torch
import torch.nn as nn
from time import sleep
import copy
from torchvision import transforms
from scipy.optimize import fsolve, root
from torch.utils import data
import torch.nn.functional as F
import argparse
import random
from scipy.stats import multivariate_normal
from tqdm import tqdm
import cleanlab
from load_data import Kvasir_SEG_trnDataSet, Kvasir_SEG_tstDataSet, CVC_ClinicDB_trnDataSet, CVC_ClinicDB_tstDataSet \
    , CVC_ClinicDB_noiseDataSet
import network
from network.utils import PolynomialLR
import loss
import matplotlib.pyplot as plt
from train import setup_model
from load_data import TwoStreamBatchSampler
from utils import ramps, losses

img_dir = './dataset/images/'
lbl_dir = './dataset/using_masks/'
lbl_clean_dir = './dataset/Ground Truth/'
train_list = './dataset/train_list.txt'
training_list = './dataset/training_pred/training_list.txt'
test_list = './dataset/test_list.txt'

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 50.0)

def ensemble_loss(model, train_ids, clean_ids, noise_ids):
    print("ensembling loss")
    dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_dir, list_path=train_list)
    dataloader = data.DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
    loss_ensemble_self = []
    name_ensemble = []
    loss_ensemble = []
    for step_t, (volume_batch, label_batch, name) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            volume_batch = volume_batch.float().cuda()
            label_batch = label_batch.long().cuda()
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            preds = outputs_soft[:, 1, :, :] > 0.5
            loss_list_tensor = loss.myDiceLoss(outputs, preds.long(), p=2, smooth=1e-8)
            loss_ensemble_self.append(loss_list_tensor.cpu().numpy())
            loss_list_tensor = loss.myDiceLoss(outputs, label_batch.long(), p=2, smooth=1e-8)
            loss_ensemble.append(loss_list_tensor.cpu().numpy())
            name_ensemble.append(name[0])
    loss_ensemble_self = np.asarray(loss_ensemble_self)
    loss_ensemble = np.asarray(loss_ensemble)

    clean_idxs = []
    noise_idxs = []
    clean_judge = []

    hard_ids_em = loss_ensemble_self >= 0.03
    assert name_ensemble == train_ids
    for i in range(len(train_ids)):
        if set(clean_ids).intersection([train_ids[i]]):
            clean_idxs.append(i)
            clean_judge.append(True)
        if set(noise_ids).intersection([train_ids[i]]):
            noise_idxs.append(i)
            clean_judge.append(False)
    clean_judge = np.asarray(clean_judge)
    clean_judge = clean_judge[~hard_ids_em]
    T_s = loss_ensemble_self[~hard_ids_em][~clean_judge].mean()/2 + \
                  loss_ensemble_self[~hard_ids_em][clean_judge].mean()/2
    T_l = loss_ensemble[~hard_ids_em][~clean_judge].mean()/2 + \
                  loss_ensemble[~hard_ids_em][clean_judge].mean()/2
    correct_index = 1 * ~hard_ids_em
    #breakpoint()
    for i in range(len(loss_ensemble_self)):
        if loss_ensemble_self[i] < T_s:
            if correct_index[i]:
                correct_index[i] = 2  #pred instead noisy lbl
        elif loss_ensemble[i] < T_l:
            if correct_index[i]:
                correct_index[i] = 1   #lbl instead noisy lbl
        else:
            if correct_index[i]:
                correct_index[i] = -1  #unsupervise
    return loss_ensemble_self, loss_ensemble, name_ensemble, correct_index, clean_idxs, noise_idxs

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def update_ema_variables(model, ema_model, global_step, alpha=0.99):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_model_state_dict = ema_model.state_dict()
        for entry in ema_model_state_dict.keys():
            ema_param = ema_model_state_dict[entry].clone().detach()
            param = model_state_dict[entry].clone().detach()
            new_param = (ema_param * alpha) + (param * (1. - alpha))
            ema_model_state_dict[entry] = new_param
        ema_model.load_state_dict(ema_model_state_dict)
    return alpha


def validating(model, img_dir, lbl_clean_dir, test_list, dataset, device):
    model.eval()
    print('testing')
    with torch.no_grad():
        total_dice = 0
        tst_dataset = CVC_ClinicDB_tstDataSet(img_dir, lbl_clean_dir, test_list, means=dataset.means,
                                              stdevs=dataset.stdevs)
        tstloader = data.DataLoader(tst_dataset, batch_size=1, num_workers=2, shuffle=False)
        for step_v, (v_x, v_y, name) in enumerate(tstloader):
            v_x = v_x.float().to(device)
            # v_y = v_y.long().to(device)
            tst_outputs = model(v_x, turnoff_drop=True)
            tst_outputs = nn.functional.softmax(tst_outputs, dim=1)
            preds = np.asarray(np.argmax(tst_outputs.detach().cpu().numpy(), axis=1), dtype=np.uint8)
            # preds = preds.transpose(1, 2, 0)
            val_dice = loss.calculate_dice(preds[0], v_y[0])
            # disc_mIOU, cup_mIOU = loss.calculate_mIOU(preds, v_y[0])
            total_dice += val_dice

        print("test_dice=%f" % (total_dice / len(tstloader)))


def cl_correction(outputs_soft, label_batch,
                  uncertainty, volume_batch, ema_model, gt, hard_ids,
                  loss_list, loss_list_self, name):
    num_classes = 2
    batchsize = 16
    T_s = loss_list_self[~hard_ids].mean()
    T_l = loss_list[~hard_ids].mean()
    correct_index = 1 * ~hard_ids
    for i in range(batchsize):
        if loss_list_self[i] < T_s:
            if correct_index[i]:
                correct_index[i] = 2  #pred instead noisy lbl
        elif loss_list[i] < T_l:
            if correct_index[i]:
                correct_index[i] = 1   #lbl instead noisy lbl
        else:
            if correct_index[i]:
                correct_index[i] = -1  #unsupervise

        #breakpoint()


        # uncertainty_np = uncertainty.cpu().detach().numpy()
        # uncertainty_np_squeeze = np.squeeze(uncertainty_np)
        # smooth_arg = 1 - uncertainty_np_squeeze
        # new_c = np.zeros_like(outputs_soft.detach().cpu())
        # new_c[:, 1, :, :] = outputs_soft[:, 1, :, :].detach().cpu().numpy() * smooth_arg
        # new_c[:, 0, :, :] = 1 - new_c[:, 1, :, :]
        # new_c_p = new_c[:, 1, :, :] > 0.5
        # for i in range(batchsize):
        #     plt.subplot(2, 2, 1)
        #     plt.imshow(outputs_soft[i, 1, :, :].detach().cpu())
        #     plt.axis('off')
        #     plt.title('outputs_soft')
        #     plt.subplot(2, 2, 2)
        #     plt.imshow(uncertainty[i, 0, :, :].detach().cpu())
        #     plt.axis('off')
        #     plt.title('uncertainty')
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(outputs_soft[i, 1, :, :].detach().cpu() > 0.5)
        #     plt.axis('off')
        #     plt.title('pred')
        #     plt.subplot(2, 2, 4)
        #     plt.imshow(gt[i,:,:].detach().cpu())
        #     plt.axis('off')
        #     plt.title('gt')
        #     plt.show()
        # breakpoint()
        #
        # CL_inputs = volume_batch.detach()  # + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)r
        # out_main = ema_model(CL_inputs)
        # pred_soft_np = torch.softmax(out_main, dim=1).cpu().detach().numpy()
        # masks_np = label_batch[:].cpu().detach().numpy()
        # preds_softmax_np_accumulated = np.swapaxes(pred_soft_np, 1, 2)
        # preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
        # preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, num_classes)
        # preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)
        # masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)
        # noise = cleanlab.filter.find_label_issues(masks_np_accumulated,
        #                                           preds_softmax_np_accumulated,
        #                                           filter_by='both', n_jobs=1)
        # confident_maps_np = noise.reshape(-1, 256, 256).astype(np.uint8)
        # corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * (corrected_masks_np)
        #
        # noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_soft.device.index)
        #
        # for i in range(batchsize):
        #     # pred_soft_np = torch.softmax(out_main[i].unsqueeze(dim=0), dim=1).cpu().detach().numpy()
        #     # masks_np = label_batch[i].unsqueeze(dim=0).cpu().detach().numpy()
        #     # preds_softmax_np_accumulated = np.swapaxes(pred_soft_np, 1, 2)
        #     # preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
        #     # preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, num_classes)
        #     # preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)
        #     # masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)
        #     # noise = cleanlab.filter.find_label_issues(masks_np_accumulated,
        #     #                                           preds_softmax_np_accumulated,
        #     #                                           filter_by='both', n_jobs=1)
        #     # confident_maps_np = noise.reshape(-1, 256, 256).astype(np.uint8)
        #     # corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
        #     plt.subplot(2, 2, 1)
        #     plt.imshow(gt[i, :, :].detach().cpu())
        #     plt.title('gt')
        #     plt.axis('off')
        #     plt.subplot(2, 2, 2)
        #     plt.imshow(label_batch[i, :, :].detach().cpu())
        #     plt.title('lbl')
        #     plt.axis('off')
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(corrected_masks_np[i, :, :])
        #     plt.title('corrected_masks')
        #     plt.axis('off')
        #     plt.subplot(2, 2, 4)
        #     plt.imshow(pred_soft_np[i, 1, :, :]>0.5)
        #     plt.axis('off')
        #     plt.title('outputs_soft')
        #     plt.show()

    return correct_index


def mean_teacher_train():
    train_ids = [i_id.strip() for i_id in open(train_list)]
    clean_ids = [i_id.strip() for i_id in open(training_list)]
    noise_ids = list(set(train_ids).difference(set(clean_ids)))
    if os.path.exists('./dataset/noise_list.txt'):
        os.remove('./dataset/noise_list.txt')
    file = open('./dataset/noise_list.txt', 'a')
    for noise_id in list(noise_ids):
        file.write(noise_id + '\n')
    file.close()
    ##setup gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    flag = torch.cuda.is_available()
    print("GPU availabel:", flag)
    device = torch.device('cuda')
    print(torch.cuda.device_count())
    print("Device: %s" % device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())
    torch.manual_seed(1)  # cpu
    torch.cuda.manual_seed(1)  # gpu
    np.random.seed(1)  # numpy
    random.seed(1)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

    def create_model(pretrained=False, dropout=False):
        # setup model
        model1, optimizer1, scheduler1 = setup_model(device, name='resnet101', has_dropout=dropout)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=500, gamma=0.6)
        if pretrained:
            pred_net = './dataset/High_noise/result0.5/clean_selected.pth'
            checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
            model1.module.load_state_dict(checkpoint["model_state"])
            print("Model restored from %s" % pred_net)
        # if ema:
        #     for param in model1.parameters():
        #         param.detach_()
        return model1, optimizer1, scheduler1

    model, optimizer, scheduler = create_model(pretrained=True, dropout=False)
    ema_model, _, _ = create_model(pretrained=True, dropout=True)
    model = model.module
    ema_model = ema_model.module
    model.cuda()
    ema_model.cuda()
    # breakpoint()
    # ema_model.eval()
    consistency_type = 'mse'
    if consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    # model = nn.DataParallel(model.cuda())
    # ema_model = nn.DataParallel(ema_model.cuda())
    loss_ensemble_self, loss_ensemble, name_ensemble, correct_index, clean_idxs, noise_idxs = \
        ensemble_loss(model, train_ids, clean_ids, noise_ids)



    dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_dir, list_path=train_list)
    dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir, list_path=train_list,
                                        means=dataset.means,
                                        stdevs=dataset.stdevs)
    batchsize = 16
    noise_batchsize = batchsize - int(batchsize * len(clean_idxs) / (len(clean_idxs) + len(noise_idxs)))
    clean_batchsize = batchsize - noise_batchsize
    batch_sampler = TwoStreamBatchSampler(clean_idxs, noise_idxs, batchsize, noise_batchsize)
    dataloader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    print("clean/noise batchsize: %d, %d" % (clean_batchsize, noise_batchsize))
    maxepoch = 20
    iter_num = 0
    num_classes = 2
    focal_loss = loss.FocalLoss()
    Correction = False
    # validating(model, img_dir, lbl_clean_dir, test_list, dataset, device)
    for epoch in range(maxepoch):
        print("\n**************Epoch=%d**********" % epoch)

        if iter_num >= -500000:
            Correction = True
        torch.cuda.empty_cache()
        # model.eval()
        # loss_ensemble_self, loss_ensemble, name_ensemble, correct_index, clean_idxs, noise_idxs = \
        #     ensemble_loss(model, train_ids, clean_ids, noise_ids)
        model.train()
        print("train...,lr1 = %f, lr2 = %f" % (optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
        pbar = tqdm(enumerate(dataloader))
        for step_t, (volume_batch, label_batch, gt, name) in pbar:
            # print("\nstep: %d" % step_t)
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            labeled_volume_batch = volume_batch[:clean_batchsize]
            # unlabeled_volume_batch = volume_batch[0].unsqueeze(0)
            # for i in range(batchsize-1):
            #     if correct_index[train_ids.index(name[i])] == 2:
            #         labeled_volume_batch = torch.cat((labeled_volume_batch, volume_batch[i+1].unsqueeze(0)))
            #     if correct_index[train_ids.index(name[i])] == 1:
            #         labeled_volume_batch = torch.cat((labeled_volume_batch, volume_batch[i + 1].unsqueeze(0)))
            # unlabeled_volume_batch = volume_batch
            #
            # noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            #
            # ema_inputs = unlabeled_volume_batch + noise
            #breakpoint()
            # ema_inputs = volume_batch + torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            # print(volume_batch.shape)
            outputs = model(labeled_volume_batch)
            # print("\niter_num: %d" % iter_num)
            #
            # outputs_soft = torch.softmax(outputs, dim=1)
            #
            # with torch.no_grad():
            #     ema_output = ema_model(ema_inputs)
            #     ema_output_soft = torch.softmax(ema_output, dim=1)
            #     preds = outputs_soft[:, 1, :, :] > 0.5
            #     loss_list_self = np.zeros(len(label_batch))
            #     loss_list = np.zeros(len(label_batch))
            #     model.eval()
            #     for i in range(len(label_batch)):
            #         loss_list_tensor = loss.myDiceLoss(outputs.detach()[i].unsqueeze(0),
            #                                            preds[i].unsqueeze(0).long(), p=2, smooth=1e-8)
            #         loss_list_self[i] = loss_list_tensor.cpu().numpy()
            #         loss_list_tensor = loss.myDiceLoss(outputs.detach()[i].unsqueeze(0),
            #                                            label_batch[i].unsqueeze(0).long(), p=2, smooth=1e-8)
            #         loss_list[i] = loss_list_tensor.cpu().numpy()
            #     model.train()
            #     hard_ids = loss_list_self > 0.05
            #     # hard_ids = hard_ids[clean_batchsize:]
            #     # Uncertainty Estimate
            #     T = 8
            #     _, _, w, h = unlabeled_volume_batch.shape
            #     volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            #     stride = volume_batch_r.shape[0] // 2
            #     preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
            #     for i in range(T // 2):
            #         ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
            #         with torch.no_grad():
            #             preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            #     preds = F.softmax(preds, dim=1)
            #     preds = preds.reshape(T, stride, num_classes, w, h)
            #     preds = torch.mean(preds, dim=0)
            #     uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            #     uncertainty = uncertainty / np.log(2)  # normalize uncertainty, cuz ln2 is the max value
            #
            #     if Correction:
            #         correct_index = cl_correction(outputs_soft.detach(), label_batch,
            #                                       uncertainty, volume_batch, ema_model, gt, hard_ids,
            #                                       loss_list, loss_list_self, name)
            #         corrected_label_batch = torch.zeros_like(label_batch[0])
            #         corrected_label_batch = torch.unsqueeze(corrected_label_batch, dim=0)
            #         preds = outputs_soft[:, 1, :, :] > 0.5
            #         for i in range(len(correct_index)):
            #             if i==0:
            #                 corrected_label_batch[0] = preds[0]
            #             else:
            #                 if correct_index[i] == 0:
            #                     corrected_label_batch = torch.cat((corrected_label_batch, label_batch[i].unsqueeze(0)))
            #                 if correct_index[i] == 1:
            #                     corrected_label_batch = torch.cat((corrected_label_batch, label_batch[i].unsqueeze(0)))
            #                 if correct_index[i] == 2:
            #                     corrected_label_batch = torch.cat((corrected_label_batch, preds[i].unsqueeze(0)))
            #         #breakpoint()
            # supervised_loss
            loss_CrossEntropy = loss.myCrossEntropy2d(outputs, label_batch[:clean_batchsize].long())
            DiceLoss = loss.myDiceLoss(outputs, label_batch[:clean_batchsize].long(), p=2, smooth=1e-8)
            # # consistency_loss
            # consistency_weight = get_current_consistency_weight(iter_num // 30)
            # consistency_dist = consistency_criterion(outputs, ema_output)  # (batch, 2, 112,112,80)
            # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, rampup_length=1550))
            # mask = (uncertainty < threshold).float()
            # # breakpoint()
            # mask = mask[~hard_ids]
            # consistency_dist = consistency_dist[~hard_ids]
            # consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            # consistency_loss = 20 * consistency_weight * consistency_dist
            # backward
            total_loss = loss_CrossEntropy + DiceLoss #+ consistency_loss
            #optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            alpha = update_ema_variables(model, ema_model, global_step=iter_num)
            iter_num = iter_num + 1
            pbar.set_description(
                "iter_num=%d, Loss_total = %f"#, consistency_loss = %f"  # , Focal_loss = %f"#Smoothloss = %f"
                % (iter_num, total_loss))#, consistency_loss))  # , focal_loss))
        if Correction:
            print('UDS correct the noisy label')
        validating(model, img_dir, lbl_clean_dir, test_list, dataset, device)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, './dataset/clean_pretrained.pth')
        # torch.save({
        #     "model_state": ema_model.state_dict(),
        #     "optimizer_state": optimizer.state_dict(),
        #     "scheduler_state": scheduler.state_dict(),
        # }, './dataset/teacher.pth')





if __name__ == '__main__':
    mean_teacher_train()
