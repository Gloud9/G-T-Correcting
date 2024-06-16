
import torch.nn.functional as F

from glob import glob
import matplotlib.pyplot as plt
import argparse
import os
import random
import shutil
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
import loss
from torch.utils import data
from MTCL import losses, ramps
from load_data import Kvasir_SEG_trnDataSet, Kvasir_SEG_tstDataSet, CVC_ClinicDB_trnDataSet, CVC_ClinicDB_tstDataSet\
    , CVC_ClinicDB_noiseDataSet

import network
from network.utils import PolynomialLR
# HD loss and boundary loss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from load_data import TwoStreamBatchSampler
# CL
import cleanlab


img_dir = './dataset/images/'
lbl_dir = './dataset/using_masks/'
lbl_clean_dir = './dataset/Ground Truth/'
train_list = './dataset/train_list.txt'

test_list = './dataset/test_list.txt'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/IRCAD_c', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='IRCAD_c/MTCL_UDS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=2000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[2, 320, 320],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1000, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--gpu', type=str, default='2',
                    help='gpu id')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# pretrain
parser.add_argument('--pretrain_model', type=str, default=None, help='pretrained model')

# CL
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')
parser.add_argument('--weak_weight', type=float,
                    default=5.0, help='weak_weight')
args = parser.parse_args()

def setup_model(device, has_dropout=False):
    # =====  Set up model  ==========  Set up model  ==========  Set up model  ==========  Set up model  ==========  Set up model  =====
    # (all models are 'constructed at network.modeling)
    CKPT_PATH = "./network/best_deeplabv3plus_resnet101_voc_os16.pth"
    model1 = network.modeling.deeplabv3plus_resnet101(num_classes=21, output_stride=8, has_dropout=has_dropout)
    model1.load_state_dict(torch.load(CKPT_PATH)['model_state'])
    model = network.modeling.deeplabv3plus_resnet101(num_classes=2, output_stride=8, has_dropout=has_dropout)
    model1.classifier = model.classifier
    del model
    model1.classifier = network.convert_to_separable_conv(model1.classifier)
    model1.apply(network.utils.freeze_bn)

    # network.utils.set_bn_momentum(model2.backbone, momentum=0.01)
    model1 = model1.to(device)

    lr1 = 0.0006
    optimizer1 = torch.optim.SGD(params=[
        {'params': model1.backbone.parameters(), 'lr': lr1},
        {'params': model1.classifier.parameters(), 'lr':  0.1 * lr1},
    ], lr=lr1, momentum=0.9, weight_decay=0.0005)
    optimizer1.param_groups[0]['momentum'] = 0.99
    optimizer1.param_groups[1]['momentum'] = 0.95

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=300, gamma=0.6)
    # scheduler1 = PolynomialLR(optimizer=optimizer1,
    #                           step_size=1,
    #                           iter_max=20000,
    #                           power=0.9)
    return model1, optimizer1, scheduler1
# BD and HD loss
def compute_dtm01(img_gt, out_shape):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """
    normalized_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):
        # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis / np.max(posdis)

    return normalized_dtm

def calculate_dice(pred_mask, true_mask):
    """Dice loss
    Args:
        predict: A tensor of shape [ *]
        target: A tensor of shape [*]
    Return:
        C dice
    """
    H=true_mask.shape[0]
    W=true_mask.shape[1]
    pred = np.zeros((H, W))
    true = np.zeros((H, W))
    # disc_dice
    true[true_mask == 0] = 0
    true[true_mask == 1] = 1
    pred[pred_mask == 0] = 0
    pred[pred_mask == 1] = 1
    dice=dice_coef(true, pred)

    return dice

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    #return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """

    img_gt = img_gt.astype(np.uint8)

    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):
        # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                        np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][c] = sdf
    return normalized_sdf


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:, 1, ...]
    dc = gt_sdf[:, 1, ...]
    multipled = torch.mul(pc, dc)
    bd_loss = multipled.mean()
    return bd_loss
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

    breakpoint()
    confidence = loss_ensemble_self[~hard_ids_em]
    confidence = confidence[clean_judge]
    confidence = (confidence - confidence.min())/(confidence.max()-confidence.min())
    confidence = 1 - confidence
    return loss_ensemble_self, loss_ensemble, name_ensemble, confidence, clean_idxs, noise_idxs
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
            val_dice = calculate_dice(preds[0], v_y[0])
            # disc_mIOU, cup_mIOU = loss.calculate_mIOU(preds, v_y[0])
            total_dice += val_dice

        print("test_dice=%f" % (total_dice / len(tstloader)))
def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    delta_s = (seg_soft[:, 1, ...] - gt.float()) ** 2
    s_dtm = seg_dtm[:, 1, ...] ** 2
    g_dtm = gt_dtm[:, 1, ...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.mul(delta_s, dtm)
    hd_loss = multipled.mean()
    return hd_loss

def labeled_slices(dataset, patiens_num):
    ref_dict = None
    if "IRCAD" in dataset:  # 1-1298 are IRCAD, others are MSD
        ref_dict = {"10": 1298}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

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
        model1, optimizer1, scheduler1 = setup_model(device, has_dropout=dropout)
        optimizer1.param_groups[0]['lr'] = args.base_lr
        optimizer1.param_groups[1]['lr'] = args.base_lr * 0.1
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=500, gamma=0.6)

        if pretrained:
            pred_net = pretrained
            checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
            model1.load_state_dict(checkpoint["model_state"])
            print("Model restored from %s" % pred_net)
        # if ema:
        #     for param in model1.parameters():
        #         param.detach_()
        return model1, optimizer1, scheduler1

    num_classes=2
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    focal_loss = losses.FocalLoss()
    model, optimizer, scheduler = create_model(pretrained=noise_dir+'clean_selected.pth', dropout=False)
    ema_model, _, _ = create_model(pretrained=noise_dir+'clean_selected.pth', dropout=True)
    clean_model, _, _ = create_model(pretrained=noise_dir+'clean_selected.pth', dropout=False)
    clean_model.cuda()
    model.cuda()
    ema_model.cuda()
    # breakpoint()
    loss_ensemble_self, loss_ensemble, name_ensemble, confidence, clean_idxs, noise_idxs = \
        ensemble_loss(clean_model, train_ids, clean_ids, noise_ids)
    breakpoint()
    dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_dir, list_path=train_list)
    dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir, list_path=train_list,
                                        means=dataset.means,
                                        stdevs=dataset.stdevs)
    batchsize = 16
    noise_batchsize = batchsize - int(batchsize * len(clean_idxs) / (len(clean_idxs) + len(noise_idxs)))
    clean_batchsize = batchsize - noise_batchsize
    batch_sampler = TwoStreamBatchSampler(clean_idxs, noise_idxs, batchsize, noise_batchsize)
    dataloader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)
    labeled_bs = clean_batchsize
    print("clean/noise batchsize: %d, %d" % (clean_batchsize, noise_batchsize))
    iter_num = 0
    model.train()
    ema_model.train()
    validating(model, img_dir, lbl_clean_dir, test_list, dataset, device)
    weak_supervised_loss = 0
    for epoch in range(100):
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
            unlabeled_volume_batch = volume_batch[clean_batchsize:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                # Uncertainty Estimate
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
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


            # breakpoint()
            # supervised_loss

            loss_ce = ce_loss(outputs[:labeled_bs], label_batch[:][:labeled_bs].long())
            loss_dice = dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))

            # focal loss
            loss_focal = focal_loss(outputs[:labeled_bs], label_batch[:][:labeled_bs].long())

            # boundary loss
            with torch.no_grad():
                gt_sdf_npy = compute_sdf1_1(label_batch.cpu().numpy(), outputs_soft.shape)
                gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(outputs_soft.device.index)
            loss_bd = boundary_loss(outputs_soft[:labeled_bs], gt_sdf[:labeled_bs])

            supervised_loss = 0.5 * (loss_ce + loss_dice) + loss_focal + 0.5 * loss_bd
            #breakpoint()
            # L_supervised_loss
            noisy_label_batch = label_batch[clean_batchsize:]
            CL_inputs = unlabeled_volume_batch
            if iter_num < 200:
                loss_ce_weak = 0.0
            elif iter_num >= 200:
                with torch.no_grad():
                    out_main = ema_model(CL_inputs)
                    pred_soft_np = torch.softmax(out_main, dim=1).cpu().detach().numpy()

                masks_np = noisy_label_batch.cpu().detach().numpy()

                preds_softmax_np_accumulated = np.swapaxes(pred_soft_np, 1, 2)
                preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
                preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, num_classes)
                preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)
                masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)

                assert masks_np_accumulated.shape[0] == preds_softmax_np_accumulated.shape[0]

                CL_type = 'both'

                try:
                    if CL_type in ['both']:

                        noise = cleanlab.filter.find_label_issues(masks_np_accumulated,
                                                          preds_softmax_np_accumulated,
                                                          filter_by=CL_type, n_jobs=1)
                    elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                        noise = cleanlab.filter.find_label_issues(masks_np_accumulated,
                                                          preds_softmax_np_accumulated,
                                                          filter_by=CL_type, n_jobs=1)
                    confident_maps_np = noise.reshape(-1, 256, 256).astype(np.uint8)

                    # label Refinement
                    correct_type = 'uncertainty_smooth'
                    if correct_type == 'fixed_smooth':
                        smooth_arg = 0.8
                        corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                        print('FS correct the noisy label')
                    elif correct_type == 'uncertainty_smooth':
                        uncertainty_np = uncertainty.cpu().detach().numpy()
                        uncertainty_np_squeeze = np.squeeze(uncertainty_np)
                        smooth_arg = 1 - uncertainty_np_squeeze
                        corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                        print('UDS correct the noisy label')
                    else:
                        corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
                        print('Hard correct the noisy label')
                    noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_soft.device.index)
                    loss_ce_weak = ce_loss(outputs[labeled_bs:], noisy_label_batch.long())
                    loss_focal_weak = focal_loss(outputs[labeled_bs:], noisy_label_batch.long())
                    weak_supervised_loss = 0.5 * (loss_ce_weak + loss_focal_weak)
                except Exception as e:
                    print('cannot identify errors')


            # consistency_loss

            # Unsupervised Consistency Loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss = torch.mean((outputs_soft[labeled_bs:] - ema_output_soft) ** 2)

            # Total Loss = H_Supervised + Consistency + L_supervised
            loss = supervised_loss + consistency_weight * (consistency_loss + args.weak_weight * weak_supervised_loss)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            optimizer.param_groups[0]['lr'] = lr_
            optimizer.param_groups[1]['lr'] = 0.1 * lr_


            iter_num = iter_num + 1
            pbar.set_description(
                "step_t=%d,lossC=%f,DLoss=%f,ConLoss=%f"  # , Focal_loss = %f"#Smoothloss = %f"
                % (step_t, supervised_loss, consistency_weight, weak_supervised_loss))  # , focal_loss))







        validating(model, img_dir, lbl_clean_dir, test_list, dataset, device)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, noise_dir+'REFUGE2.pth')


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
