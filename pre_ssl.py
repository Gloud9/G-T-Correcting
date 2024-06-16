import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import os
import numpy as np
import random
from tqdm import tqdm
import shutil
from glob import glob
from load_data import Kvasir_SEG_trnDataSet, Kvasir_SEG_tstDataSet, CVC_ClinicDB_trnDataSet, CVC_ClinicDB_tstDataSet \
    , CVC_ClinicDB_noiseDataSet

import loss
import matplotlib.pyplot as plt
from train import setup_model
from load_data import TwoStreamBatchSampler
from utils import ramps, losses

img_dir = './dataset/images/'
lbl_dir = './dataset/using_masks/'
lbl_clean_dir = './dataset/Ground Truth/'
train_list = './dataset/train_list.txt'

test_list = './dataset/test_list.txt'
consistency_criterion = losses.softmax_mse_loss
def mycopyfile(srcfile, dstpath, is_print=False):  # 复制函数
    # srcfile 需要复制、移动的文件
    # dstpath 目的地址
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        if is_print:
            print("copy %s -> %s" % (srcfile, dstpath + fname))
def ensemble_loss(model, train_ids, clean_ids, noise_ids):
    print("ensembling loss")
    dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_dir, list_path=train_list)
    dataloader = data.DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
    loss_ensemble_self = []
    name_ensemble = []
    loss_ensemble = []
    empty_lbl = []
    model.eval()
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
            if torch.sum(label_batch) <= 10:
                empty_lbl.append(name[0])
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
    #breakpoint()
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
    correct_index[clean_idxs] = 3
#    breakpoint()
    if len(empty_lbl):
        for name in empty_lbl:
            correct_index[name_ensemble.index(name)] = -1
    return loss_ensemble_self, loss_ensemble, name_ensemble, correct_index, clean_idxs, noise_idxs
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
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 50.0)
def mean_teacher_train(noise_dir):
    training_list = noise_dir+'training_list.txt'
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

    def create_model(pretrained, dropout=False):
        # setup model
        model1, optimizer1, scheduler1 = setup_model(device, name='resnet101', has_dropout=dropout)
        optimizer1.param_groups[0]['lr'] = 0.0006
        optimizer1.param_groups[1]['lr'] = 0.00006
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=500, gamma=0.6)
        if pretrained:
            pred_net = pretrained
            #pred_net = './dataset/student.pth'
            checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
            model1.module.load_state_dict(checkpoint["model_state"])
            print("Model restored from %s" % pred_net)
        # if ema:
        #     for param in model1.parameters():
        #         param.detach_()
        return model1, optimizer1, scheduler1


    model, optimizer, scheduler = create_model(pretrained=noise_dir+'clean_selected.pth', dropout=False)
    ema_model, _, _ = create_model(pretrained=noise_dir+'clean_selected.pth', dropout=True)
    clean_model, _, _ = create_model(pretrained=noise_dir+'clean_selected.pth', dropout=False)
    clean_model = clean_model.module
    model = model.module
    ema_model = ema_model.module

    clean_model.cuda()
    model.cuda()
    ema_model.cuda()
    # breakpoint()
    loss_ensemble_self, loss_ensemble, name_ensemble, correct_index, clean_idxs, noise_idxs = \
        ensemble_loss(clean_model, train_ids, clean_ids, noise_ids)

    dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_dir, list_path=train_list)
    dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir, list_path=train_list,
                                        means=dataset.means,
                                        stdevs=dataset.stdevs)
    batchsize = 8
    noise_batchsize = batchsize - int(batchsize * len(clean_idxs) / (len(clean_idxs) + len(noise_idxs)))
    clean_batchsize = batchsize - noise_batchsize
    batch_sampler = TwoStreamBatchSampler(clean_idxs, noise_idxs, batchsize, noise_batchsize)
    dataloader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    print("clean/noise batchsize: %d, %d" % (clean_batchsize, noise_batchsize))
    maxepoch = 50
    iter_num = 0
    num_classes = 2
    focal_loss = loss.FocalLoss()
    Correction = False
    model.train()
    ema_model.train()
    clean_model.eval()
    validating(model, img_dir, lbl_clean_dir, test_list, dataset, device)

    for epoch in range(maxepoch):
        print("\n**************Epoch=%d**********" % epoch)
        torch.cuda.empty_cache()
        # if epoch > -50:
        #     print("ensembling loss")
        #     _, _, _, correct_index, _, _ = ensemble_loss(ema_model, train_ids, clean_ids, noise_ids)
        model.train()
        ema_model.train()
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
            with torch.no_grad():
                ema_model.eval()
                # if epoch > 50:
                #     outputs = ema_model(volume_batch)
                # else:
                outputs = clean_model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                preds = outputs_soft.detach()[:, 1, :, :] > 0.5
                if epoch <= 10:

                    labeled_volume_batch = volume_batch[:clean_batchsize]
                    label_ids = list(range(clean_batchsize))
                else:
                    labeled_volume_batch = volume_batch[0].unsqueeze(0)
                    unlabeled_volume_batch = volume_batch[0].unsqueeze(0)
                    label_ids = []
                    label_batch[0] = preds[0].detach()
                    unlabel_ids = []
                    label_ids.append(0)
                    for j in range(batchsize-1):
                        i = j + 1
                        #breakpoint()
                        if correct_index[train_ids.index(name[i])] == 2:
                            labeled_volume_batch = torch.cat((labeled_volume_batch, volume_batch[i].unsqueeze(0)))
                            label_batch[i] = preds[i].detach()
                            label_ids.append(i)
                        if correct_index[train_ids.index(name[i])] == 3:
                            labeled_volume_batch = torch.cat((labeled_volume_batch, volume_batch[i].unsqueeze(0)))
                            label_ids.append(i)
                        if correct_index[train_ids.index(name[i])] == 1:
                            labeled_volume_batch = torch.cat((labeled_volume_batch, volume_batch[i].unsqueeze(0)))
                            label_ids.append(i)
                        if correct_index[train_ids.index(name[i])] == 0:
                            labeled_volume_batch = torch.cat((labeled_volume_batch, volume_batch[i].unsqueeze(0)))
                            label_ids.append(i)
                            # if len(unlabel_ids) == 0:
                            #     unlabeled_volume_batch = volume_batch[i].unsqueeze(0)
                            #     unlabel_ids.append(i)
                            # else:
                            #     unlabeled_volume_batch = torch.cat((unlabeled_volume_batch, volume_batch[i].unsqueeze(0)))
                            #     unlabel_ids.append(i)
                        if correct_index[train_ids.index(name[i])] == -1:
                            if len(unlabel_ids) == 0:
                                unlabeled_volume_batch = volume_batch[i].unsqueeze(0)
                                unlabel_ids.append(i)
                            else:
                                unlabeled_volume_batch = torch.cat((unlabeled_volume_batch, volume_batch[i].unsqueeze(0)))
                                unlabel_ids.append(i)
            with torch.no_grad():
                # Uncertainty Estimate
                ema_model.train()
                T = 4
                _, _, w, h = volume_batch.shape
                volume_batch_r = volume_batch[clean_batchsize:].repeat(2, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
                for i in range(T // 2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.3, 0.3)
                    with torch.no_grad():
                        preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, num_classes, w, h)
                preds = torch.mean(preds, dim=0)
                uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
                uncertainty = uncertainty / np.log(2)
                noise = torch.clamp(torch.randn_like(volume_batch[clean_batchsize:]) * 0.1, -0.3, 0.3)
                ema_output = ema_model(volume_batch[clean_batchsize:] + noise)
            # breakpoint()
            # supervised_loss
            optimizer.zero_grad()

            outputs = model(volume_batch)
            breakpoint()
            # if epoch == 15:
            #     breakpoint()
            loss_CrossEntropy = loss.myCrossEntropy2d(outputs[label_ids], label_batch[label_ids].long())
            DiceLoss = 20 * loss.myDiceLoss(outputs[label_ids], label_batch[label_ids].long(), p=2, smooth=1e-8)
            #breakpoint()
            # consistency_loss
            consistency_weight = get_current_consistency_weight(iter_num // 30)
            consistency_dist = consistency_criterion(outputs[clean_batchsize:], ema_output)  # (batch, 2, 112,112,80)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, rampup_length=1550))
            mask = (uncertainty < threshold).float()
            consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            consistency_loss = 20 * consistency_weight * consistency_dist
            # for i in range(batchsize):
            #     plt.subplot(2, 2, 1)
            #     plt.imshow(outputs_soft[i, 1, :, :].detach().cpu())
            #     plt.axis('off')
            #     plt.title('outputs_soft')
            #     plt.subplot(2, 2, 2)
            #     plt.imshow(label_batch[i , :, :].detach().cpu())
            #     plt.axis('off')
            #     plt.title('label')
            #     plt.subplot(2, 2, 3)
            #     plt.imshow(outputs_soft[i, 1, :, :].detach().cpu() > 0.5)
            #     plt.axis('off')
            #     plt.title('pred')
            #     plt.subplot(2, 2, 4)
            #     plt.imshow(gt[i,:,:].detach().cpu())
            #     plt.axis('off')
            #     plt.title('gt')
            #     plt.show()
            # print('loss_list_self:', loss_list_self)
            # print('Ts: ', loss_list_self[~hard_ids].mean())
            # print('loss_list:', loss_list)
            # print('Tl: ', loss_list[~hard_ids].mean())
            # breakpoint()
            # backward

            total_loss = loss_CrossEntropy + DiceLoss + consistency_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            alpha = update_ema_variables(model, ema_model, global_step=iter_num)
            iter_num = iter_num + 1
            pbar.set_description(
                "step_t=%d,lossC=%f,DLoss=%f,ConLoss=%f"  # , Focal_loss = %f"#Smoothloss = %f"
                % (step_t, loss_CrossEntropy, DiceLoss, consistency_loss))  # , focal_loss))






        #
        # validating(model, img_dir, lbl_clean_dir, test_list, dataset, device)
        # torch.save({
        #     "model_state": model.state_dict(),
        #     "optimizer_state": optimizer.state_dict(),
        #     "scheduler_state": scheduler.state_dict(),
        # }, noise_dir+' all_c.pth')
        # torch.save({
        #     "model_state": ema_model.state_dict(),
        #     "optimizer_state": optimizer.state_dict(),
        #     "scheduler_state": scheduler.state_dict(),
        # }, './dataset/teacher.pth')

if __name__ == '__main__':
    print("copying dataset")
    src_dir = './dataset/Ground Truth/'
    dst_dir = './dataset/using_masks/'  # 目的路径记得加斜杠
    #breakpoint()
    src_file_list = glob(src_dir + '*.tif')  # glob获得路径下所有文件，可根据需要修改
    for i in range(len(src_file_list)):
        mycopyfile(src_file_list[i], dst_dir, is_print=False)  # 复制文件
        if src_file_list[i].startswith(src_dir.rstrip('/') + '\\'):
            src_file_list[i] = src_file_list[i][len(src_dir.rstrip('/') + '\\'):]
    noise_dir = './dataset/High_noise/rate0.5/'
    #noise_dir = './dataset/corrected_masks/high0.5/'
    dst_dir = './dataset/using_masks/'  # 目的路径记得加斜杠
    noise_file_list = glob(noise_dir + '*.tif')  # glob获得路径下所有文件，可根据需要修改
    for i in range(len(noise_file_list)):
        mycopyfile(noise_file_list[i], dst_dir, is_print=True)  # 复制文件
        if noise_file_list[i].startswith(noise_dir.rstrip('/')):
            noise_file_list[i] = noise_file_list[i][len(noise_dir.rstrip('/') + '\\'):]
        print(noise_file_list[i])

    t_list = [i_id.strip() for i_id in open('./dataset/train_list.txt')]
    clean_file_list = list(set(t_list).difference(set(noise_file_list)))
    #breakpoint()
    if os.path.exists('./dataset/training_pred/clean_list.txt'):
        os.remove('./dataset/training_pred/clean_list.txt')
    file = open('./dataset/training_pred/clean_list.txt', 'a')
    for str in clean_file_list:
        file.write(str + '\n')
    file.close()
    if os.path.exists('./dataset/training_pred/noise_list.txt'):
        os.remove('./dataset/training_pred/noise_list.txt')
    file = open('./dataset/training_pred/noise_list.txt', 'a')
    for str in noise_file_list:
        file.write(str + '\n')
    file.close()
    #breakpoint()
    mean_teacher_train(noise_dir.replace('rate','result'))
