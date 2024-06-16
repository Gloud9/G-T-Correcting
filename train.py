import os
import shutil
from glob import glob
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import copy
from scipy.optimize import fsolve, root
from torch.utils import data
import torch.nn.functional as F
import argparse
from scipy.stats import multivariate_normal
from tqdm import tqdm
from load_data import Kvasir_SEG_trnDataSet, Kvasir_SEG_tstDataSet, CVC_ClinicDB_trnDataSet, CVC_ClinicDB_tstDataSet\
    , CVC_ClinicDB_noiseDataSet
import network
from network.utils import PolynomialLR
import loss
import matplotlib.pyplot as plt
import scipy.io as scio


def gy_GMM(X, rate=0.1):#rate: noise rate
    rate = 1 - rate
    k = 2
    N = len(X)
    EPS = 0.001
    C = rate
    M = 1 - rate
    Miu = np.random.rand(k, 1)
    Miu[0] = -0.5
    Miu[1] = 0.5
    Posterior = np.zeros((N, k))
    sigma = np.random.rand(k, 1)
    # sigma[0] = 1
    # sigma[1] = 1
    alpha = np.random.rand(k, 1)
    alpha[0] = C
    alpha[1] = M
    #rate = 0.1
    rate = rate / (1 - rate)
    Posterior = np.zeros((N, k))

    # print(sigma)
    for it in range(1000):
        # 先求后验概率
        for i in range(N):
            dominator = 0
            for j in range(k):
                dominator = dominator + np.exp(-1.0 / (2.0 * sigma[j]) * (X[i] - Miu[j]) ** 2)
            # print -1.0/(2.0*sigma[j]),(X[i] - Miu[j])**2,-1.0/(2.0*sigma[j]) * (X[i] - Miu[j])**2,np.exp(-1.0/(2.0*sigma[j]) * (X[i] - Miu[j])**2)
            # return
            for j in range(k):
                numerator = np.exp(-1.0 / (2.0 * sigma[j]) * (X[i] - Miu[j]) ** 2)
                Posterior[i, j] = numerator / dominator
        oldMiu = copy.deepcopy(Miu)
        oldalpha = copy.deepcopy(alpha)
        oldsigma = copy.deepcopy(sigma)
        # 最大化
        dominator = np.zeros(2)
        numerator = np.zeros(2)
        doubletor = np.zeros(2)
        for j in range(k):  # dominator:Nk  numerator:sum(gama(n,k)*xn))
            for i in range(N):
                numerator[j] = numerator[j] + Posterior[i, j] * X[i]
                dominator[j] = dominator[j] + Posterior[i, j]
            Miu[j] = numerator[j] / dominator[j]
            alpha[j] = dominator[j] / N
            tmp = 0
            for i in range(N):
                tmp = tmp + Posterior[i, j] * (X[i] - Miu[j]) ** 2
                doubletor[j] = doubletor[j] + Posterior[i, j] * X[i] * X[i]
            sigma[j] = tmp / dominator[j]

        # def f(X):
        #     miu, s0, s1 = X[0], X[1], X[2]
        #     return [(miu * (s1 * dominator[0] + s0 * dominator[1]) + s0 * numerator[1] - s1 * numerator[0]),
        #             (s0 * dominator[0] - doubletor[0] + 2 * miu * numerator[0] - dominator[0] * miu * miu),
        #             (s1 * dominator[1] - doubletor[1] - 2 * miu * numerator[1] - dominator[1] * miu * miu)]

        def f(X):
            miu, s0, s1 = X[0], X[1], X[2]
            return [(miu * (s1 * dominator[0] + s0 * dominator[1] * rate) + s0 * numerator[1] - s1 * numerator[0]),
                    (s0 * dominator[0] - doubletor[0] + 2 * miu * numerator[0] - dominator[0] * miu * miu),
                    (s1 * dominator[1] - doubletor[1] - 2 * miu * numerator[1] * rate - dominator[1] * miu * miu * rate * rate)]
        X0 = np.hstack((Miu[0], sigma[0], sigma[1])).reshape(-1, 1)
        miu, s0, s1 = fsolve(f, X0)

        Miu[0] = miu
        Miu[1] = miu * C/(C-1)
        sigma[0] = s0
        sigma[1] = s1
        # alpha[0] = 0.5
        # alpha[1] = 0.5
        if ((abs(Miu - oldMiu)).sum() < EPS) and \
                ((abs(alpha - oldalpha)).sum() < EPS) and \
                ((abs(sigma - oldsigma)).sum() < EPS):
            # print('it: ', it)
            # print('Miu: %f, %f' %(Miu[0],Miu[1]))
            # print('sigma: %f, %f' %(sigma[0],sigma[1]))
            # print('alpha: %f, %f' %(alpha[0],alpha[1]))
            return Miu, np.sqrt(sigma), alpha, it

    return Miu, np.sqrt(sigma), alpha, it

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

def new_train_list(loss_ensemble, pred, name_ensemble):
    if os.path.exists('./dataset/training_pred/training_list.txt'):
        os.remove('./dataset/training_pred/training_list.txt')
    file = open('./dataset/training_pred/training_list.txt', 'a')
    for i in range(len(name_ensemble)):
        if pred[i] == 1:
            file.write(name_ensemble[i] + '\n')
    file.close()
    return './dataset/training_pred/training_list.txt'


def setup_model(device, name='resnet101', has_dropout=False):
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
    # network.utils.set_bn_momentum(model1.backbone, momentum=0.01)

    CKPT_PATH = "./network/best_deeplabv3plus_mobilenet_voc_os16.pth"
    model2 = network.modeling.deeplabv3plus_mobilenet(num_classes=21, output_stride=8)
    model2.load_state_dict(torch.load(CKPT_PATH)['model_state'])
    model = network.modeling.deeplabv3plus_mobilenet(num_classes=2, output_stride=8)
    model2.classifier = model.classifier
    del model
    model2.classifier = network.convert_to_separable_conv(model2.classifier)
    model2.apply(network.utils.freeze_bn)
    # network.utils.set_bn_momentum(model2.backbone, momentum=0.01)
    model1 = model1.to(device)
    model2 = model2.to(device)

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
    lr2 = 0.0006
    optimizer2 = torch.optim.SGD(params=[
        {'params': model2.backbone.parameters(), 'lr': lr2},
        {'params': model2.classifier.parameters(), 'lr': 0.1 * lr2},
    ], lr=lr2, momentum=0.9, weight_decay=0.0005)
    optimizer2.param_groups[0]['momentum'] = 0.99
    optimizer2.param_groups[1]['momentum'] = 0.95
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=200, gamma=0.6)
    # scheduler2 = PolynomialLR(optimizer=optimizer2,
    #                           step_size=1,
    #                           iter_max=160000,
    #                           power=0.9)
    model1 = nn.DataParallel(model1)
    model2 = nn.DataParallel(model2)
    if name == 'resnet101':
        return model1, optimizer1, scheduler1
    if name == 'mobilenet':
        return model2, optimizer2, scheduler2


def train(model_path, pred_net=None):
    d = 'dataset'
    clean_file_list = './dataset/training_pred/clean_list.txt'
    noise_file_list = './dataset/training_pred/noise_list.txt'
    if d == 'dataset_K':
        img_dir = './dataset_K/images/'
        lbl_dir = './dataset_K/using_masks/'
        lbl_clean_dir = './dataset_K/masks/'
        train_list = './dataset_K/train_list.txt'

        val_list = './dataset_K/val_list.txt'
        test_list = './dataset_K/test_list.txt'
    else:
        img_dir = './dataset/images/'
        lbl_dir = './dataset/using_masks/'
        lbl_clean_dir = './dataset/Ground Truth/'
        train_list = './dataset/train_list.txt'
        training_list = train_list
        val_list = './dataset/test_list.txt'
        test_list = './dataset/test_list.txt'
    # setup gpu
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
    # setup dataset
    train_dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_dir, training_list)
    trainloader = data.DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True)
    trainset_means = train_dataset.means
    trainset_std = train_dataset.stdevs


    tst_dataset = CVC_ClinicDB_tstDataSet(img_dir, lbl_clean_dir, val_list, means=trainset_means,
                                          stdevs=trainset_std)
    tstloader = data.DataLoader(tst_dataset, batch_size=1, num_workers=2, shuffle=False)

    noise_dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir, list_path=noise_file_list, means=trainset_means,
                                            stdevs=trainset_std)
    noiseloader = data.DataLoader(noise_dataset, batch_size=1, num_workers=2, shuffle=False)

    clean_dataset = CVC_ClinicDB_tstDataSet(img_dir, lbl_dir, clean_file_list, means=trainset_means,
                                            stdevs=trainset_std)
    cleanloader = data.DataLoader(clean_dataset, batch_size=1, num_workers=2, shuffle=False)

    train_ensemble_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=2, shuffle=False)
    #breakpoint()
    # train_dataset = Kvasir_SEG_trnDataSet(img_dir, lbl_dir, train_list)
    # trainloader = data.DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True, )
    # val_dataset = Kvasir_SEG_tstDataSet(img_dir, lbl_clean_dir, val_list, means=train_dataset.means, stdevs=train_dataset.stdevs)
    # valloader = data.DataLoader(val_dataset, batch_size=1, num_workers=2)
    print("Train dataset:", len(train_dataset))
    print("Tst dataset:", len(tstloader))

    # setup model
    model1, optimizer1, scheduler1 = setup_model(device, name='resnet101')
    if not pred_net == None:
        checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
        model1.module.load_state_dict(checkpoint["model_state"])
        model1.to(device)
        model1.cuda()
        print("Model restored from %s" %pred_net)
    model1 = model1.module



    max_epoch = 50
    loss_list_sum = []
    loss_list_ce = []
    loss_list_bsc = []
    loss_list_smth = []
    val_list_dice = []

    # Focal_loss = loss.FocalLoss()
    clean_dice = np.zeros((len(clean_dataset), max_epoch), dtype=np.float32)
    noise_dice = np.zeros((len(noise_dataset), max_epoch), dtype=np.float32)
    noise_gt_dice = np.zeros((len(noise_dataset), max_epoch), dtype=np.float32)
    clean_uncertainty1 = np.zeros((len(clean_dataset), max_epoch), dtype=np.float32)
    noise_uncertainty1 = np.zeros((len(noise_dataset), max_epoch), dtype=np.float32)
    clean_uncertainty2 = np.zeros((len(clean_dataset), max_epoch), dtype=np.float32)
    noise_uncertainty2 = np.zeros((len(noise_dataset), max_epoch), dtype=np.float32)

    clean_loss = np.zeros((len(clean_dataset), max_epoch), dtype=np.float32)
    noise_loss = np.zeros((len(noise_dataset), max_epoch), dtype=np.float32)
    clean_name = []
    noise_name = []
    clean_select_flag = 0
    iter_num = 0
    for epoch in range(max_epoch):

        model1.train()
        # if epoch == 50:
        #     optimizer1.param_groups[0]['lr'] = 0.0006
        #     optimizer1.param_groups[1]['lr'] = 0.00006
        print("\n**************Epoch=%d**********" % epoch)
        train_dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_dir, training_list, means=trainset_means, stdevs=trainset_std)
        trainloader = data.DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True)
        # model1.apply(network.utils.freeze_bn)
        train_iter = iter(trainloader)
        print("train...,lr1 = %f, lr2 = %f" % (optimizer1.param_groups[0]['lr'], optimizer1.param_groups[1]['lr']))
        #breakpoint()
        pbar = tqdm(range(len(trainloader)))
        for step_t in pbar:

            t_x, t_y, name = train_iter.next()
            # if t_x.shape[0] < 2:
            #     step_t = step_t-1
            #     continue
            if t_x.shape[0] < 2:
                t_x = torch.vstack((t_x, t_x))
                t_y = torch.vstack((t_y, t_y))

            t_x = t_x.float().to(device)
            t_y = t_y.long().to(device)
            optimizer1.zero_grad()
            outputs1 = model1(t_x)  # N,C,H,W
            loss_CrossEntropy = loss.myCrossEntropy2d(outputs1, t_y)
            DiceLoss = 20 * loss.myDiceLoss(outputs1, t_y, p=2, smooth=1e-8)
            # focal_loss = 5 * Focal_loss(outputs1, t_y)
            #Smoothloss = 30 * loss.Smoothloss(outputs1, t_y)
            Loss1 = loss_CrossEntropy + DiceLoss #+ Smoothloss# + focal_loss#+ Smoothloss
            pbar.set_description(
                "Loss_total = %f, CrossEntropy = %f, DiceLoss = %f"  # , Focal_loss = %f"#Smoothloss = %f"
                % (Loss1, loss_CrossEntropy, DiceLoss))  # , focal_loss))
            Loss1.backward()
            optimizer1.step()
            scheduler1.step()
            iter_num += 1
            loss_list_sum.append(Loss1.detach().cpu().numpy())
            loss_list_ce.append(loss_CrossEntropy.detach().cpu().numpy())
            loss_list_bsc.append(DiceLoss.detach().cpu().numpy())
            loss_list_smth.append(DiceLoss.detach().cpu().numpy())
        # validation
        print('loss_sum:',
              np.mean(loss_list_sum[(iter_num - step_t):iter_num]))
        print('loss_ce:',
              np.mean(loss_list_ce[(iter_num - step_t):iter_num]))
        print('loss_bsc:',
              np.mean(loss_list_bsc[(iter_num - step_t):iter_num]))
        print('loss_smooth:',
              np.mean(loss_list_smth[(iter_num - step_t):iter_num]))
        with torch.no_grad():
            print('testing')
            model1.eval()
            total_dice = 0
            #breakpoint()
            for step_v, (v_x, v_y, name) in enumerate(tstloader):
                v_x = v_x.float().to(device)
                # v_y = v_y.long().to(device)
                outputs = model1(v_x)
                outputs = nn.functional.softmax(outputs, dim=1)
                preds = np.asarray(np.argmax(outputs.detach().cpu().numpy(), axis=1), dtype=np.uint8)
                # preds = preds.transpose(1, 2, 0)
                val_dice = loss.calculate_dice(preds[0], v_y[0])
                # disc_mIOU, cup_mIOU = loss.calculate_mIOU(preds, v_y[0])
                total_dice += val_dice

            print("test_dice=%f" % (total_dice / len(tstloader)))
            val_list_dice.append(total_dice / len(tstloader))
            torch.save({
                "model_state": model1.state_dict(),
                "optimizer_state": optimizer1.state_dict(),
                "scheduler_state": scheduler1.state_dict(),
                "Loss_sum": loss_list_sum,
                "Loss_ce": loss_list_ce,
                "Loss_bsc": loss_list_bsc,
                "Loss_smth": loss_list_smth,
                "val_dice": val_list_dice,
            }, model_path)
            print("model saved at %s" % (model_path))
            if not os.path.exists(os.path.join('./dataset/training_pred/epoch/', 'epoch%d/' % (epoch))):
                os.makedirs(os.path.join('./dataset/training_pred/epoch/', 'epoch%d' % (epoch)))
            # if not os.path.exists(os.path.join('./dataset/training_pred/epoch/', 'epoch%d/' % (epoch), 'clean')):
            #     os.makedirs(os.path.join('./dataset/training_pred/epoch/', 'epoch%d/' % (epoch), 'clean'))
            # if not os.path.exists(os.path.join('./dataset/training_pred/epoch/', 'epoch%d/' % (epoch), 'noise')):
            #     os.makedirs(os.path.join('./dataset/training_pred/epoch/', 'epoch%d/' % (epoch), 'noise'))

            clean_name = []
            noise_name = []
            for step_t, (t_x, t_y, name) in tqdm(enumerate(cleanloader)):
                t_x = t_x.float().to(device)
                outputs = model1(t_x)
                t_y = t_y.long().to(device)
                loss_CrossEntropy = loss.myCrossEntropy2d(outputs.detach(), t_y)
                DiceLoss = 20 * loss.myDiceLoss(outputs.detach(), t_y, p=2, smooth=1e-8)

                clean_loss[step_t][epoch] = loss_CrossEntropy.detach().cpu().numpy() + 0.1 * DiceLoss.detach().cpu().numpy()
                clean_name.append(name[0])
                preds = np.asarray(np.argmax(outputs.detach().cpu().numpy(), axis=1), dtype=np.uint8)
                val_dice = loss.calculate_dice(preds[0], t_y[0].detach().cpu().numpy())
                clean_dice[step_t][epoch] = val_dice
                # outputs = outputs[0][1].detach().cpu().numpy()
                # t_y = t_y[0].detach().cpu().numpy()
                # clean_uncertainty1[step_t][epoch], clean_uncertainty2[step_t][epoch] =\
                #     loss.calclulate_uncertainty(outputs, t_y)
                # outputs = outputs * 255
                # t_y = t_y * 255
                # img = np.hstack((outputs, t_y))
                # plt.imshow(np.hstack((outputs, t_y)))
                # plt.savefig(os.path.join('./dataset/training_pred/epoch/', 'epoch%d/'% (epoch), 'clean/', name[0].replace('.tif', '.jpg')))
                # plt.cla()
                # plt.clf()
            for step_t, (t_x, t_y, gt, name) in tqdm(enumerate(noiseloader)):
                t_x = t_x.float().to(device)
                outputs = model1(t_x)
                t_y = t_y.long().to(device)
                loss_CrossEntropy = loss.myCrossEntropy2d(outputs.detach(), t_y)
                DiceLoss = 20 * loss.myDiceLoss(outputs.detach(), t_y, p=2, smooth=1e-8)

                noise_loss[step_t][epoch] = loss_CrossEntropy.detach().cpu().numpy() + 0.1 * DiceLoss.detach().cpu().numpy()
                noise_name.append(name[0])
                preds = np.asarray(np.argmax(outputs.detach().cpu().numpy(), axis=1), dtype=np.uint8)
                # preds = preds.transpose(1, 2, 0)
                val_dice = loss.calculate_dice(preds[0], t_y[0].detach().cpu().numpy())
                noise_dice[step_t][epoch] = val_dice
                val_dice_gt = loss.calculate_dice(preds[0], gt[0])
                noise_gt_dice[step_t][epoch] = val_dice_gt
                # outputs = outputs[0][1].detach().cpu().numpy()
                #
                # t_y = t_y[0].detach().cpu().numpy()
                # noise_uncertainty1[step_t][epoch], noise_uncertainty2[step_t][epoch] = \
                     #loss.calclulate_uncertainty(outputs, t_y)
                # outputs = outputs * 255
                # t_y = t_y * 255
                #img = np.hstack((outputs, t_y))
                # plt.imshow(np.hstack((outputs, t_y)))
                # plt.savefig(os.path.join('./dataset/training_pred/epoch/', 'epoch%d/'% (epoch), 'noise/', name[0].replace('.tif', '.jpg')))
                # plt.cla()
                # plt.clf()
            train_dice = np.vstack((clean_dice, noise_dice))
            plt.plot(train_dice[:,epoch], 'b.')
            plt.title("epoch%d:total/clean/noise/noise_gt dice = %.3f,%.3f,%.3f,%.3f"%(epoch, train_dice[:, epoch].mean(),
                                                                                       clean_dice[:, epoch].mean(), noise_dice[:, epoch].mean(), noise_gt_dice[:, epoch].mean()))
            plt.show()
            plt.savefig(os.path.join(('./dataset/training_pred/epoch/epoch%d/散点图.jpg'%epoch)))
            plt.clf()
            plt.cla()
            # GMM
            if epoch >= 10105:
                loss_ensemble = np.hstack((clean_loss[:, epoch], noise_loss[:, epoch]))
                clean_num = clean_loss.shape[0]
                noise_num = noise_loss.shape[0]

                name_ensemble = clean_name + noise_name
                pred = np.zeros((len(train_ensemble_loader), 1))
                # for step_v, (v_x, v_y, name) in enumerate(train_ensemble_loader):
                #     outputs = model1(v_x)
                #     v_y = v_y.long().to(device)
                #     loss_CrossEntropy = loss.myCrossEntropy2d(outputs.detach(), v_y)
                #     DiceLoss = 20 * loss.myDiceLoss(outputs.detach(), v_y, p=2, smooth=1e-8)
                #     loss_ensemble[step_v] = loss_CrossEntropy.detach().cpu().numpy() + 0.1 * DiceLoss.detach().cpu().numpy()
                #     name_ensemble.append(name[0])
                x = loss_ensemble.reshape(-1, 1)
                x = (x-x.mean())/x.std()
                #breakpoint()
                warnings.filterwarnings('error')
                try:
                    Miu, sigma, alpha, it = gy_GMM(x)
                except RuntimeWarning as e:
                    print(e)
                    print('retry gmm')
                    try:
                        Miu, sigma, alpha, it = gy_GMM(x)
                    except RuntimeWarning as e:
                        print(e)
                        print('retry gmm')
                        Miu, sigma, alpha, it = gy_GMM(x)
                #Miu, sigma, alpha, it = gy_GMM(x)
                warnings.filterwarnings('default')
                mu1, mu2 = Miu[0], Miu[1]
                print("it:%d" % it)
                print("mu1, mu2: %.3f, %.3f" % (mu1, mu2))
                if min(mu1, mu2) <= -0.99 and clean_select_flag==0:
                    print("change gmm")
                    clean_select_flag = 1
                    #continue
                if it > 500:
                    print("bad gmm")
                    continue


                #breakpoint()
                sigma1, sigma2 = sigma[0], sigma[1]
                alpha1, alpha2 = alpha[0], alpha[1]
                norm1 = multivariate_normal(mu1, sigma1)  # 输入的是方差
                norm2 = multivariate_normal(mu2, sigma2)
                tau1 = alpha1 * norm1.pdf(x)
                tau2 = alpha2 * norm2.pdf(x)
                line = np.arange(-2, 4, 0.0001)
                #plt.figure()

                if mu1 < mu2:
                    tau = tau1
                else:
                    tau = tau2
                tau = tau / (tau1 + tau2)
                tau = tau.reshape((-1,1))
                tau = tau / tau.max()
                pred[x <= min(mu1, mu2)] = 1
                pred[x >= max(mu1, mu2)] = 0
                pred[(x > min(mu1, mu2)) * (x < max(mu1, mu2))] = tau[((x > min(mu1, mu2)) * (x < max(mu1, mu2)))]

                if epoch >= 50 or clean_select_flag:
                    if clean_select_flag == 0:
                        optimizer1.param_groups[0]['lr'] = 0.0006
                        optimizer1.param_groups[1]['lr'] = 0.00006
                    clean_select_flag = 1
                    pred = pred > 0.5

                if epoch >=10:
                    pred = pred > 0.5
                print(sum(pred == 1))
                print('clean_select_flag', clean_select_flag)
                plt.subplot(221)
                plt.plot(line, alpha1 * norm1.pdf(line) + alpha2 * norm2.pdf(line), 'b')
                if mu1 < mu2:
                    plt.plot(line, alpha1 * norm1.pdf(line), 'g')
                    plt.plot(line, alpha2 * norm2.pdf(line), 'r')
                else:
                    plt.plot(line, alpha1 * norm1.pdf(line), 'r')
                    plt.plot(line, alpha2 * norm2.pdf(line), 'g')

                plt.subplot(222)
                plt.title(epoch)
                plt.plot(tau, 'b.')
                plt.subplot(223)
                plt.plot(np.arange(0, clean_num, 1), x[0:clean_num], 'b.')
                plt.plot(np.arange(clean_num, noise_num+clean_num, 1), x[clean_num:noise_num+clean_num], 'r.')
                plt.subplot(224)
                #breakpoint()
                plt.hist(x[0:clean_num], bins=150, rwidth=1, color='green')
                plt.hist(x[clean_num:490], bins=150, rwidth=1, color='red')
                # _ = plt.hist(np.hstack((x[0:441], x[441:490])), bins=150, rwidth=1, range=(-1.5, 1.5), align='left',
                #              color=['green', 'red'])

                plt.show()
                plt.savefig(os.path.join(('./dataset/training_pred/epoch/epoch%d/分布图.jpg'%epoch)))
                plt.clf()
                plt.cla()
                training_list = new_train_list(loss_ensemble, pred, name_ensemble)
                #breakpoint()



    # scio.savemat('./dataset/training_pred/clean_loss/res_high0.5.mat',
    #              {'clean_loss': clean_loss, 'clean_uncertainty1': clean_uncertainty1, 'clean_uncertainty2': clean_uncertainty2})
    # scio.savemat('./dataset/training_pred/noisy_loss/res_high0.5.mat',
    #              {'noise_loss': clean_loss, 'noisen_uncertainty1': noise_uncertainty1, 'noise_uncertainty2': noise_uncertainty2})
    breakpoint()
    plt.plot(np.mean(train_dice,axis=0), 'black')
    plt.plot(np.mean(clean_dice, axis=0), 'y')
    plt.plot(np.mean(noise_dice, axis=0), 'r')
    plt.plot(np.mean(noise_gt_dice, axis=0), 'g')
    plt.grid()
    plt.show()
    plt.savefig('./dataset/training_pred/epoch/训练历史图.jpg')
    plt.clf()
    plt.cla()
    shutil.copy('./dataset/training_pred/training_list.txt', model_path.replace('clean_selected.pth', 'training_list.txt'))

    gt_list = [i_id.strip() for i_id in open('./dataset/training_pred/clean_list.txt')]
    noise_list = [i_id.strip() for i_id in open('./dataset/training_pred/noise_list.txt')]
    clean_list = [i_id.strip() for i_id in open('./dataset/training_pred/training_list.txt')]
    a = len(set(gt_list).intersection(clean_list))
    b = len(set(clean_list).intersection(noise_list))
    c = len(gt_list)-a
    d = len(noise_list)-b
    print(a,b)
    print(c,d)
    print("acc=",(a+d)/(c+b+a+d))
    shutil.move('./dataset/training_pred/epoch/', model_path.replace('clean_selected.pth', '/'))
    breakpoint()




if __name__ == '__main__':#改gmm比例，文件比例，结果目录比例
    print("copying dataset")
    src_dir = './dataset/Ground Truth/'
    dst_dir = './dataset/using_masks/'  # 目的路径记得加斜杠
    #breakpoint()
    src_file_list = glob(src_dir + '*.tif')  # glob获得路径下所有文件，可根据需要修改
    for i in range(len(src_file_list)):
        mycopyfile(src_file_list[i], dst_dir, is_print=False)  # 复制文件
        if src_file_list[i].startswith(src_dir.rstrip('/') + '\\'):
            src_file_list[i] = src_file_list[i][len(src_dir.rstrip('/') + '\\'):]
    noise_dir = './dataset/Low_noise/rate0.3/'
    #noise_dir = './dataset/corrected_masks/high0.5/'
    dst_dir = './dataset/using_masks/'  # 目的路径记得加斜杠
    noise_file_list = glob(noise_dir + '*.tif')  # glob获得路径下所有文件，可根据需要修改
    for i in range(len(noise_file_list)):
        mycopyfile(noise_file_list[i], dst_dir, is_print=True)  # 复制文件
        if noise_file_list[i].startswith(noise_dir.rstrip('/')):
            noise_file_list[i] = noise_file_list[i][len(noise_dir.rstrip('/') + '\\'):]
        print(noise_file_list[i])

    train_list = [i_id.strip() for i_id in open('./dataset/train_list.txt')]
    clean_file_list = list(set(train_list).difference(set(noise_file_list)))
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
    # resample train/test set
    # img_dir = './dataset_K/images/'
    # lbl_dir = './dataset_K/using_masks/'
    # filelist = os.listdir(img_dir)
    # random.shuffle(filelist)
    # train_list = filelist[0:int(0.8 * len(filelist))]
    # val_list = filelist[int(0.8 * len(filelist)):int(0.9 * len(filelist))]
    # test_list = filelist[int(0.9 * len(filelist)):len(filelist)]
    # # train_list = list(set(filelist).difference(test_list))
    # if os.path.exists('./dataset_K/train_list.txt'):
    #     os.remove('./dataset_K/train_list.txt')
    # file = open('./dataset_K/train_list.txt', 'a')
    # for str in train_list:
    #     file.write(str + '\n')
    # file.close()
    # if os.path.exists('./dataset_K/val_list.txt'):
    #     os.remove('./dataset_K/val_list.txt')
    # file = open('./dataset_K/val_list.txt', 'a')
    # for str in val_list:
    #     file.write(str + '\n')
    # file.close()
    # if os.path.exists('./dataset_K/test_list.txt'):
    #     os.remove('./dataset_K/test_list.txt')
    # file = open('./dataset_K/test_list.txt', 'a')
    # for str in test_list:
    #     file.write(str + '\n')
    # file.close()
    #
    # gt_list = [i_id.strip() for i_id in open('./dataset/training_pred/clean_list.txt')]
    # noise_list = [i_id.strip() for i_id in open('./dataset/training_pred/noise_list.txt')]
    # clean_list = [i_id.strip() for i_id in open('./dataset/High_noise/result0.5/training_list.txt')]
    # a = len(set(gt_list).intersection(clean_list))
    # b = len(set(clean_list).intersection(noise_list))
    # c = len(gt_list)-a
    # d = len(noise_list)-b
    # print(a,b)
    # print(c,d)
    # print("acc=",(a+d)/(c+b+a+d))
    # # breakpoint()
    train(model_path='./dataset/Low_noise/result0.3/resnet101.pth')#pred_net='./dataset/cl_resnet101.pth'
