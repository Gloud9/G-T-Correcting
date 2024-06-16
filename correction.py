import numpy as np
import torch
import os
import warnings

warnings.simplefilter('ignore', ResourceWarning)
warnings.simplefilter('ignore', DeprecationWarning)
import torch.nn as nn
from glob import glob
import math
from PIL import Image
from tqdm import tqdm
import copy
from torch.utils import data
from scipy.optimize import fsolve, root
from load_data import CVC_ClinicDB_trnDataSet, CVC_ClinicDB_tstDataSet, CVC_ClinicDB_noiseDataSet
from train import setup_model
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import shutil
from train import mycopyfile
import cleanlab
import loss


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def select_correction(noise_uncertainty, noise_name, correction_rate=0.5):
    hard_id = []
    hard_name = []
    easy_id = []
    easy_name = []
    for i in range(len(noise_uncertainty)):
        if noise_uncertainty[i] >= 50:
            hard_id.append(noise_uncertainty[i])
            hard_name.append(noise_name[i])
        else:
            easy_id.append(noise_uncertainty[i])
            easy_name.append(noise_name[i])

    treshold = np.quantile(easy_id, correction_rate)

    #treshold = np.mean(easy_id)
    print('correction treshold:%.3f' % treshold)
    correction_id = []
    correction_name = []
    maintain_id = []
    maintain_name = []
    for i in range(len(easy_id)):
        if easy_id[i] > treshold:
            maintain_id.append(easy_id[i])
            maintain_name.append(easy_name[i])
        else:
            correction_id.append(easy_id[i])
            correction_name.append(easy_name[i])
    return hard_id, hard_name, maintain_id, maintain_name, correction_id, correction_name


def select_correction2(noise_uncertainty, noise_name, correction_rate=0.5):
    hard_name = []
    easy_name = []
    hard_id = np.asarray(noise_uncertainty)[np.asarray(noise_uncertainty) >= 100]
    easy_id = np.asarray(noise_uncertainty)[np.asarray(noise_uncertainty) < 100]
    for i in range(len(noise_name)):
        if (np.asarray(noise_uncertainty) >= 100)[i]:
            hard_name.append(noise_name[i])
        else:
            easy_name.append(noise_name[i])
    x = easy_id
    a = np.quantile(x, 0.75)  # 上四分之一数
    b = np.quantile(x, 0.25)  # 下四分之一数
    print("平均数：", np.mean(x))  # 打印均值
    print("中位数：", np.median(x))  # 打印中位数
    print("上四分之一数：", a)  # 打印上四分之一数
    print("下四分之一数：", b)  # 打印下四分之一数
    up = a + 1.5 * (a - b)  # 异常值判断标准s
    down = b - 1.5 * (a - b)  # 异常值判断标准
    x = np.sort(x)  # 对原始数据排序
    shangjie = x[x < up][-1]  # 除了异常值外的最大值
    xiajie = x[x > down][0]  # 除了异常值外的最小值
    print("上界：", shangjie)  # 打印上界
    print("up:", up)
    print("down:", down)
    print("下界：", xiajie)  # 打印下界
    plt.grid(True)  # 显示网格
    y = plt.boxplot(x, meanline=True, showmeans=True,
                    flierprops={"marker": "o", "markerfacecolor": "red"})  # 绘制箱形图，设置异常点大小、样式等
    plt.show()  # 显示图
    hard_id = list(hard_id)
    easy_id = list[easy_id]
    easy_id = []
    easy_name = []
    correction_id = []
    correction_name = []
    maintain_id = []
    maintain_name = []
    for i in range(len(x)):
        if x[i] >= up:
            hard_id.append(x[i])
            hard_name.append(noise_name[i])
        if x[i] > np.mean(x):
            maintain_id.append(x[i])
            maintain_name.append(noise_name[i])
        else:
            correction_id.append(x[i])
            correction_name.append(noise_name[i])
    return hard_id, hard_name, maintain_id, maintain_name, correction_id, correction_name


def gy_GMM(X, rate):
    k = 2
    N = len(X)
    EPS = 0.001
    C = 0.5
    M = 0.5
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
                    (s1 * dominator[1] - doubletor[1] - 2 * miu * numerator[1] * rate - dominator[
                        1] * miu * miu * rate * rate)]

        X0 = np.hstack((Miu[0], sigma[0], sigma[1])).reshape(-1, 1)
        miu, s0, s1 = fsolve(f, X0)

        Miu[0] = miu
        Miu[1] = -rate * miu
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


def label_corretion(pred_net):
    torch.manual_seed(1)
    np.random.seed(1)
    ##setup gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"
    flag = torch.cuda.is_available()
    print("GPU availabel:", flag)
    device = torch.device('cuda')
    print(torch.cuda.device_count())
    print("Device: %s" % device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())
    model1, optimizer1, scheduler1 = setup_model(device, name='resnet101')

    checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
    model1.module.load_state_dict(checkpoint["model_state"])
    model1.to(device)
    model1 = nn.DataParallel(model1)
    model1.cuda()
    print("Model restored from %s" % pred_net)
    num_channels = 2
    img_dir = './dataset/images/'
    lbl_dir = './dataset/using_masks/'
    test_list = './dataset/test_list.txt'
    train_list = './dataset/train_list.txt'
    training_list = './dataset/training_pred/training_list.txt'
    lbl_clean_dir = './dataset/Ground Truth/'
    # label correction-CL
    train_dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_dir, train_list)
    noise_dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir, list_path=train_list,
                                              means=train_dataset.means,
                                              stdevs=train_dataset.stdevs)
    trainloader = data.DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True)
    tst_dataset = CVC_ClinicDB_tstDataSet(img_dir, lbl_clean_dir, test_list, means=train_dataset.means,
                                          stdevs=train_dataset.stdevs)
    tstloader = data.DataLoader(tst_dataset, batch_size=1, num_workers=2, shuffle=False)

    clean_ids = [i_id.strip() for i_id in open(training_list)]
    noise_ids = [i_id.strip() for i_id in open(train_list)]
    noise_ids = set(noise_ids).difference(set(clean_ids))
    if os.path.exists('./dataset/corrected_masks/noise_list.txt'):
        os.remove('./dataset/corrected_masks/noise_list.txt')
    file = open('./dataset/corrected_masks/noise_list.txt', 'a')
    for noise_id in list(noise_ids):
        file.write(noise_id + '\n')
    file.close()

    rate = 0.5
    model1.eval()
    pltdir = ['D:\GY\研二科研\暑假\组会\iter\it0',
              'D:\GY\研二科研\暑假\组会\iter\it1',
              'D:\GY\研二科研\暑假\组会\iter\it2',
              'D:\GY\研二科研\暑假\组会\iter\it3',
              'D:\GY\研二科研\暑假\组会\iter\it4'
              ]
    for iteration in range(6):
        # breakpoint()
        del_file('./dataset/corrected_masks/contrast')
        del_file('./dataset/corrected_masks/pred')
        print("\n**************iteration=%d**********" % iteration)
        print("loss ensembling")
        ensemble_dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir,
                                                     list_path='./dataset/corrected_masks/noise_list.txt',
                                                     means=train_dataset.means, stdevs=train_dataset.stdevs)
        ensemble_loader = data.DataLoader(ensemble_dataset, batch_size=1, num_workers=2, shuffle=False)
        model1.eval()
        uncertainty = []
        uncertainty2 = []
        name_list = []
        # correction_loss = np.zeros(len(ensemble_dataset))
        # for step_t, (t_x, t_y, gt, name) in tqdm(enumerate(ensemble_loader)):
        #     t_x = t_x.float().to(device)
        #     t_y = t_y.long().to(device)
        #     outputs = model1(t_x)  # N,C,H,W
        #     loss_CrossEntropy = loss.myCrossEntropy2d(outputs, t_y)
        #     DiceLoss = 20 * loss.myDiceLoss(outputs, t_y, p=2, smooth=1e-8)
        #     correction_loss[step_t] = loss_CrossEntropy.detach().cpu().numpy() + 0.1 * DiceLoss.detach().cpu().numpy()
        #     name_list.append(name[0])
        #
        #     outputs = nn.functional.softmax(outputs, dim=1)
        #     outputs = outputs.detach().cpu().numpy()
        #     pred = outputs[0][1] > 0.5
        #     uncertainty.append(sum(sum((~ pred) * outputs[0][1])) / (np.sqrt(sum(sum(pred))) + 0.01))
        #     pred = Image.fromarray(255 * pred.astype(np.uint8))
        #     pred.save(os.path.join('./dataset/corrected_masks/pred', name[0]))

        clean_file_list = './dataset/training_pred/clean_list.txt'
        noise_file_list = './dataset/training_pred/noise_list.txt'
        noise_dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir, list_path=noise_file_list,
                                                  means=train_dataset.means, stdevs=train_dataset.stdevs)
        noiseloader = data.DataLoader(noise_dataset, batch_size=1, num_workers=2, shuffle=False)

        clean_dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir, list_path=clean_file_list,
                                                  means=train_dataset.means, stdevs=train_dataset.stdevs)
        cleanloader = data.DataLoader(clean_dataset, batch_size=1, num_workers=2, shuffle=False)
        clean_loss = np.zeros((len(clean_dataset), 2), dtype=np.float32)
        noise_loss = np.zeros((len(noise_dataset), 2), dtype=np.float32)
        epoch = 0
        gt_name = []
        for step_t, (t_x, t_y, gt, name) in tqdm(enumerate(cleanloader)):
            t_x = t_x.float().to(device)
            outputs = model1(t_x)
            t_y = t_y.long().to(device)
            loss_CrossEntropy = loss.myCrossEntropy2d(outputs.detach(), t_y)
            DiceLoss = 20 * loss.myDiceLoss(outputs.detach(), t_y, p=2, smooth=1e-8)

            clean_loss[step_t][epoch] = loss_CrossEntropy.detach().cpu().numpy() + 0.1 * DiceLoss.detach().cpu().numpy()
            name_list.append(name[0])
            gt_name.append(name[0])
            outputs = nn.functional.softmax(outputs, dim=1)
            outputs = outputs.detach().cpu().numpy()
            pred = outputs[0][1] > 0.5
            # preds_softmax_np = outputs
            # preds_softmax_np = np.swapaxes(preds_softmax_np, 1, 2)
            # preds_softmax_np = np.swapaxes(preds_softmax_np, 2, 3)
            # preds_softmax_np = preds_softmax_np.reshape(-1, num_channels)
            # preds_softmax_np = np.ascontiguousarray(preds_softmax_np)
            # masks_np = t_y.detach().cpu().numpy().reshape(-1).astype(np.uint8)
            # noise = cleanlab.filter.find_label_issues(masks_np, preds_softmax_np, n_jobs=1)
            # confident_maps_np = noise.reshape(-1, 256, 256).astype(np.uint8) * 255

            # img_11 = 255 * outputs[0][1]
            # img_12 = 255 * gt[0].detach().cpu().numpy()
            # img_21 = 255 * t_y[0].detach().cpu().numpy()
            #
            # img_22 = confident_maps_np[0]
            # img_tmp1 = np.hstack((img_11, img_12))
            # img_tmp2 = np.hstack((img_21, img_22))
            # img = np.vstack((img_tmp1, img_tmp2))
            # plt.cla()
            # plt.clf()
            # plt.imshow(img)
            # plt.axis('off')
            # plogp = -outputs[0][1] * np.log(outputs[0][1])
            CL1 = sum(sum(outputs[0][1] * t_y[0].detach().cpu().numpy())) / (
                    sum(sum(t_y[0].detach().cpu().numpy())) + 0.01)
            # plt.title('(gt)uncertain:%.3f, loss:%.3f, CL:' % (
            #     sum(sum((~ pred) * plogp)) / (np.sqrt(sum(sum(pred))) + 0.001), clean_loss[step_t][epoch]))
            #           #sum(sum(pred * outputs[0][1])) / (sum(sum(pred)) + 0.001)))
            #
            # plt.savefig(os.path.join('./dataset/corrected_masks/contrast', name[0].replace('tif', 'jpg')))
            # plt.cla()
            # plt.clf()
            if step_t == 0:
                preds_softmax_np_accumulated = outputs
                masks_np_accumulated = t_y.detach().cpu().numpy()
            else:
                preds_softmax_np_accumulated = np.concatenate((preds_softmax_np_accumulated, outputs), axis=0)
                masks_np_accumulated = np.concatenate((masks_np_accumulated, t_y.detach().cpu().numpy()), axis=0)
            uncertainty.append(sum(sum((~ pred) * outputs[0][1])) / (np.sqrt(sum(sum(pred))) + 0.001))
            uncertainty2.append(CL1)
            # confidence.append(sum(sum(pred * outputs[0][1])) / (sum(sum(pred)) + 0.001))
            pred = Image.fromarray(255 * pred.astype(np.uint8))
            pred.save(os.path.join('./dataset/corrected_masks/pred', name[0]))

        for step_t, (t_x, t_y, gt, name) in tqdm(enumerate(noiseloader)):
            t_x = t_x.float().to(device)
            outputs = model1(t_x)
            t_y = t_y.long().to(device)
            loss_CrossEntropy = loss.myCrossEntropy2d(outputs.detach(), t_y)
            DiceLoss = 20 * loss.myDiceLoss(outputs.detach(), t_y, p=2, smooth=1e-8)

            noise_loss[step_t][epoch] = loss_CrossEntropy.detach().cpu().numpy() + 0.1 * DiceLoss.detach().cpu().numpy()


            outputs = nn.functional.softmax(outputs, dim=1)

            preds = outputs[0][1]
            plogp = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
            uncertainty.append(plogp/math.log(2))
            outputs = outputs.detach().cpu().numpy()
            pred = outputs[0][1] > 0.5
            name_list.append(name[0])
            uncertainty_dice = loss.myDiceLoss(torch.tensor(outputs).cuda(),
                                               torch.tensor(pred.reshape((1, 256, 256))).cuda(), p=2,
                                               smooth=1e-8)
            uncertainty_dice = uncertainty_dice.cpu().numpy()
            # preds_softmax_np = outputs
            # preds_softmax_np = np.swapaxes(preds_softmax_np, 1, 2)
            # preds_softmax_np = np.swapaxes(preds_softmax_np, 2, 3)
            # preds_softmax_np = preds_softmax_np.reshape(-1, num_channels)
            # preds_softmax_np = np.ascontiguousarray(preds_softmax_np)
            # masks_np = t_y.detach().cpu().numpy().reshape(-1).astype(np.uint8)
            # if len(np.unique(masks_np)) > 1:
            #     noise = cleanlab.filter.find_label_issues(masks_np, preds_softmax_np, n_jobs=1)
            #     confident_maps_np = noise.reshape(-1, 256, 256).astype(np.uint8) * 255
            #     img_22 = confident_maps_np[0]
            # else:
            #     img_22 = pred
            #
            # img_11 = 255 * outputs[0][1]
            # img_12 = 255 * gt[0].detach().cpu().numpy()
            # img_21 = 255 * t_y[0].detach().cpu().numpy()
            #
            # img_tmp1 = np.hstack((img_11, img_12))
            # img_tmp2 = np.hstack((img_21, img_22))
            # img = np.vstack((img_tmp1, img_tmp2))
            # plt.cla()
            # plt.clf()
            # plt.imshow(img)
            # plt.axis('off')

            breakpoint()
            # plt.title('uncertain:%.3f, loss:%.3f' % (
            #     sum(sum((~ pred) * plogp)) / (np.sqrt(sum(sum(pred))) + 0.001), noise_loss[step_t][epoch]))
            #
            # plt.savefig(os.path.join('./dataset/corrected_masks/contrast', name[0].replace('tif', 'jpg')))
            # plt.cla()
            # plt.clf()
            preds_softmax_np_accumulated = np.concatenate((preds_softmax_np_accumulated, outputs), axis=0)
            masks_np_accumulated = np.concatenate((masks_np_accumulated, t_y.detach().cpu().numpy()), axis=0)
            #uncertainty.append(sum(sum((~ pred) * outputs[0][1])) / (np.sqrt(sum(sum(pred))) + 0.01))
            CL1 = sum(sum(outputs[0][1] * t_y[0].detach().cpu().numpy())) / (
                    sum(sum(t_y[0].detach().cpu().numpy())) + 0.01)
            uncertainty2.append(CL1)
            # confidence.append(sum(sum(pred * outputs[0][1])) / (sum(sum(pred)) + 0.001))
            pred = Image.fromarray(255 * pred.astype(np.uint8))
            pred.save(os.path.join('./dataset/corrected_masks/pred', name[0]))

        ## GMM
        correction_loss = np.hstack((clean_loss[:, epoch], noise_loss[:, epoch]))
        x = correction_loss.reshape(-1, 1)
        x = (x - x.mean()) / x.std()
        warnings.filterwarnings('error')

        try:
            Miu, sigma, alpha, it = gy_GMM(x, rate)
        except RuntimeWarning as e:
            print(e)
            print('retry gmm')
            try:
                Miu, sigma, alpha, it = gy_GMM(x, rate)
            except RuntimeWarning as e:
                print(e)
                print('retry gmm')
                Miu, sigma, alpha, it = gy_GMM(x, rate)
        warnings.filterwarnings('default')
        warnings.simplefilter('ignore', ResourceWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        print("rate = %.2f" % rate)
        mu1, mu2 = Miu[0], Miu[1]
        sigma1, sigma2 = sigma[0], sigma[1]
        alpha1, alpha2 = alpha[0], alpha[1]
        norm1 = multivariate_normal(mu1, sigma1)  # 输入的是方差
        norm2 = multivariate_normal(mu2, sigma2)
        tau1 = alpha1 * norm1.pdf(x)
        tau2 = alpha2 * norm2.pdf(x)

        line = np.arange(-2, 2, 0.0001)
        plt.figure()
        plt.title('iteration=%d' % (iteration))
        plt.plot(line, alpha1 * norm1.pdf(line) + alpha2 * norm2.pdf(line), 'b')
        if mu1 < mu2:
            tau = tau1
            plt.plot(line, alpha1 * norm1.pdf(line), 'g')
            plt.plot(line, alpha2 * norm2.pdf(line), 'r')
        else:
            tau = tau2
            plt.plot(line, alpha1 * norm1.pdf(line), 'r')
            plt.plot(line, alpha2 * norm2.pdf(line), 'g')
        plt.show()
        plt.plot(tau, 'bo')
        plt.title("iteration:%d" % iteration)
        plt.show()
        tau = tau.reshape((-1, 1))
        tau = tau / tau.max()
        pred = tau > 0.5
        clean_names = []
        noise_ids = []
        for i in range(len(pred)):
            if pred[i]:
                clean_names.append(name_list[i])
            else:
                noise_ids.append(name_list[i])
        noise_ids = set(noise_ids)
        # breakpoint()
        # correction
        noise_uncertainty = []
        noise_name = list(noise_ids)
        for i in range(len(noise_ids)):  # 从id中找到uncertainty
            noise_uncertainty.append(uncertainty[name_list.index(list(noise_ids)[i])])
        # very hard samples:
        print("correction rate:%.3f" % (iteration / 5 + 0.2))
        _, hard_name, b, maintain_name, c, correction_name = select_correction(
            noise_uncertainty, noise_name, correction_rate=(iteration / 5 + 0.2))
        noise_contrast_loss = []
        for i in range(len(noise_name)):
            noise_contrast_loss.append(x[name_list.index(noise_name[i])])
        print(np.mean(noise_contrast_loss))
        # _, hard_name, c, correction_name = select_correction2(
        #     noise_uncertainty, noise_name)
        del_file('./dataset/corrected_masks/high0.5')
        if os.path.exists('./dataset/corrected_masks/high0.5/using_list.txt'):
            os.remove('./dataset/corrected_masks/high0.5/using_list.txt')
        file = open('./dataset/corrected_masks/high0.5/using_list.txt', 'a')
        file_list = []
        ###第二轮迭代，clean不能
        # breakpoint()
        # for name in noise_name:
        #     if noise_name.index(name) == 0:
        #         preds_softmax_np_accumulated_noise = preds_softmax_np_accumulated[name_list.index(name)].reshape(
        #             (1, 2, 256, 256))
        #         masks_np_accumulated_noise = masks_np_accumulated[name_list.index(name)].reshape((1, 256, 256))
        #     else:
        #         preds_softmax_np_accumulated_noise = np.concatenate((preds_softmax_np_accumulated_noise,
        #                                                              preds_softmax_np_accumulated[
        #                                                                  name_list.index(name)].reshape(
        #                                                                  (1, 2, 256, 256))), axis=0)
        #         masks_np_accumulated_noise = np.concatenate((masks_np_accumulated_noise,
        #                                                      masks_np_accumulated[name_list.index(name)].reshape(
        #                                                          (1, 256, 256))), axis=0)
        #
        #
        # preds_softmax_np_accumulated_noise = np.swapaxes(preds_softmax_np_accumulated_noise, 1, 2)
        # preds_softmax_np_accumulated_noise = np.swapaxes(preds_softmax_np_accumulated_noise, 2, 3)
        # preds_softmax_np_accumulated_noise = preds_softmax_np_accumulated_noise.reshape(-1, num_channels)
        # preds_softmax_np_accumulated_noise = np.ascontiguousarray(preds_softmax_np_accumulated_noise)
        # masks_np_accumulated_noise = masks_np_accumulated_noise.reshape(-1).astype(np.uint8)
        # noise = cleanlab.filter.find_label_issues(masks_np_accumulated_noise, preds_softmax_np_accumulated_noise,
        #                                           n_jobs=1)
        # confident_maps_np_noise = noise.reshape(-1, 256, 256).astype(np.uint8) * 255

        for idx in tqdm(range(len(noise_name))):
            filename = noise_name[idx]
            # confident_map_np = confident_maps_np_noise[idx]

            # lbl = Image.open(os.path.join('./dataset/using_masks', filename))
            # lbl = lbl.convert('1')
            # lbl = lbl.resize((256, 256))
            # lbl = np.asarray(lbl, np.float32)
            # lbl = lbl * 255
            # lbl = np.asarray(lbl, dtype=np.uint8)
            # gt = Image.open(os.path.join('./dataset/Ground Truth', filename))
            # gt = gt.convert('1')
            # gt = gt.resize((256, 256))
            # gt = np.asarray(gt, np.float32)
            # gt = gt * 255
            # gt = np.asarray(gt, dtype=np.uint8)
            # score_map = preds_softmax_np_accumulated[name_list.index(filename)]
            # pred = score_map[1] > 0.5
            # # img = Image.open(os.path.join('./dataset/images', filename.replace('.tif', '.jpg')))
            # # img = img.convert("RGB")
            # # img = img.resize((256, 256))
            # # img = np.asarray(img, np.uint8)
            # # cl1 = preds_softmax_np_accumulated[name_list.index(filename)].reshape((1,2,256,256))
            # # cl1 = np.swapaxes(cl1, 1, 2)
            # # cl1 = np.swapaxes(cl1, 2, 3)
            # # cl1 = cl1.reshape(-1, num_channels)
            # # cl1 = np.ascontiguousarray(cl1)
            # # masks1 = masks_np_accumulated[name_list.index(filename)].reshape((1, 256, 256))
            # # masks1 = masks1.reshape(-1).astype(np.uint8)
            # # if sum(masks1):
            # #     noise = cleanlab.filter.find_label_issues(masks1, cl1, n_jobs=1)
            # #     confident_maps1 = noise.reshape(-1, 256, 256).astype(np.uint8) * 255
            # plt.cla()
            # plt.clf()
            # plt.subplot(221)
            # plt.title('uncertain:(%.3f,%.3f) loss:%.3f' % (
            #     sum(sum((~ pred) * score_map[1])) / (np.sqrt(sum(sum(pred))) + 0.001),
            #     uncertainty2[name_list.index(filename)],
            #     correction_loss[name_list.index(filename)]))
            # plt.imshow(score_map[1])
            # plt.axis('off')
            # plt.subplot(222)
            # plt.imshow(gt)
            # plt.axis('off')
            # plt.title('gt')
            # plt.subplot(223)
            # plt.imshow(lbl)
            # plt.axis('off')
            # if len(set(gt_name).intersection([filename])):
            #     plt.title('gt')
            # else:
            #     plt.title('noisy mask')
            # plt.subplot(224)
            # plt.imshow(pred)
            # plt.axis('off')
            # plt.title('pred, iteration%d' % iteration)
            #
            # # plt.show()
            # plt.savefig(
            #     os.path.join(pltdir[iteration], 'noise_contrast', filename.replace('.tif', '.jpg')))
            # plt.cla()
            # plt.clf()
            if len(set(hard_name).intersection([filename])):
                shutil.copy(os.path.join(lbl_dir, filename),
                            os.path.join('./dataset/corrected_masks/high0.5', filename))

                file_list.append(filename)
            # if len(set(maintain_name).intersection([filename])):
            #     if correction_loss[name_list.index(filename)] <= np.mean(correction_loss):
            #         shutil.copy(os.path.join(lbl_dir, filename),
            #                     os.path.join('./dataset/corrected_masks/high0.5', filename))
            #         file.write(filename + '\n')
            if len(set(correction_name).intersection([filename])):
                # corrected_lbl = lbl / 255 - confident_map_np / 255
                # corrected_lbl = np.abs(corrected_lbl)
                # corrected_lbl = Image.fromarray(corrected_lbl * 255)
                # corrected_lbl.save(os.path.join('./dataset/corrected_masks/high0.5', filename))
                # pred = Image.fromarray(np.uint8(pred * 255))
                # pred.save(os.path.join('./dataset/corrected_masks/high0.5', filename))
                if iteration == 5:
                    shutil.copy(os.path.join(lbl_dir, filename),
                                os.path.join('./dataset/corrected_masks/high0.5', filename))

                    file_list.append(filename)
                else:
                    shutil.copy(os.path.join('./dataset/corrected_masks/pred', filename),
                            os.path.join('./dataset/corrected_masks/high0.5', filename))
                    file_list.append(filename)
        for id in tqdm(range(len(clean_names))):
            filename = clean_names[id]
            # lbl = Image.open(os.path.join('./dataset/using_masks', filename))
            # lbl = lbl.convert('1')
            # lbl = lbl.resize((256, 256))
            # lbl = np.asarray(lbl, np.float32)
            # lbl = lbl * 255
            # lbl = np.asarray(lbl, dtype=np.uint8)
            # gt = Image.open(os.path.join('./dataset/Ground Truth', filename))
            # gt = gt.convert('1')
            # gt = gt.resize((256, 256))
            # gt = np.asarray(gt, np.float32)
            # gt = gt * 255
            # gt = np.asarray(gt, dtype=np.uint8)
            # score_map = preds_softmax_np_accumulated[name_list.index(filename)]
            # pred = score_map[1] > 0.5
            # plt.cla()
            # plt.clf()
            # plt.subplot(221)
            # plt.title('uncertain:%.3f, loss:%.3f' % (
            #     sum(sum((~ pred) * score_map[1])) / (np.sqrt(sum(sum(pred))) + 0.001),
            #     uncertainty2[name_list.index(filename)]))
            # plt.imshow(score_map[1])
            # plt.axis('off')
            # plt.subplot(222)
            # plt.imshow(gt)
            # plt.axis('off')
            # plt.title('gt')
            # plt.subplot(223)
            # plt.imshow(lbl)
            # plt.axis('off')
            # if len(set(gt_name).intersection([filename])):
            #     plt.title('gt')
            # else:
            #     plt.title('noisy mask')
            # plt.subplot(224)
            # plt.imshow(pred)
            # plt.axis('off')
            # plt.title('pred, iteration%d' % iteration)
            #
            # # plt.show()
            # plt.savefig(
            #     os.path.join(pltdir[iteration], filename.replace('.tif', '.jpg')))
            # plt.cla()
            # plt.clf()

            shutil.copy(os.path.join(lbl_dir, filename),
                        os.path.join('./dataset/corrected_masks/high0.5', filename))
            # shutil.copy(os.path.join('./dataset/using_masks', filename),
            #             os.path.join('./dataset/corrected_masks/using_corrected', filename))
            file_list.append(filename)

        for id in tqdm(range(len(clean_ids))):
            filename = clean_ids[id]
            shutil.copy(os.path.join(lbl_dir, filename),
                        os.path.join('./dataset/corrected_masks/high0.5', filename))
            # shutil.copy(os.path.join('./dataset/using_masks', filename),
            #             os.path.join('./dataset/corrected_masks/using_corrected', filename))
            file_list.append(filename)
        # for name in correction_name:
        #     shutil.copy(os.path.join('./dataset/corrected_masks/pred', name),
        #                 os.path.join('./dataset/corrected_masks\high0.5', name))
        #     file.write(name + '\n')
        #     count += 1
        #     # if x[name_list.index(name)] <= 0:
        #     #     shutil.copy(os.path.join(lbl_dir, name),
        #     #                 os.path.join('./dataset/corrected_masks\high0.5', name))
        #     #    file.write(name + '\n')
        # for name in maintain_name:
        #     if correction_loss[name_list.index(name)] <= np.mean(noise_contrast_loss):
        #         shutil.copy(os.path.join(lbl_dir, name),
        #                     os.path.join('./dataset/corrected_masks\high0.5', name))
        #         file.write(name + '\n')
        #
        # for name in hard_name:
        #     shutil.copy(os.path.join(lbl_dir, name),
        #                 os.path.join('./dataset/corrected_masks\high0.5', name))
        #     file.write(name + '\n')
        for filename in set(file_list):
            file.write(filename + '\n')
        file.close()
        b.extend(c)
        plt.plot(b, 'bo')
        plt.title('iteration:%d, treshold:%.3f, mean_loss:%.3f' % (iteration, np.mean(b), np.mean(correction_loss)))
        plt.show()

        # rate = rate + (1 - rate) * (iteration / 5 + 0.2)
        # rate = (len([i_id.strip() for i_id in open('./dataset/corrected_masks/high0.5/using_list.txt')]) -len(clean_ids))/(490 - len(clean_ids))
        rate = len([i_id.strip() for i_id in open('./dataset/corrected_masks/high0.5/using_list.txt')]) / 490
        # breakpoint()
        lbl_dir = './dataset/corrected_masks/using_corrected/'
        src_dir = './dataset/using_masks/'
        src_file_list = glob(src_dir + '*.tif')  # glob获得路径下所有文件，可根据需要修改
        for i in range(len(src_file_list)):
            mycopyfile(src_file_list[i], lbl_dir, is_print=False)  # 复制文件
            if src_file_list[i].startswith(src_dir.rstrip('/') + '\\'):
                src_file_list[i] = src_file_list[i][len(src_dir.rstrip('/') + '\\'):]
        src_dir = './dataset/corrected_masks/high0.5/'
        src_file_list = glob(src_dir + '*.tif')  # glob获得路径下所有文件，可根据需要修改
        for i in range(len(src_file_list)):
            mycopyfile(src_file_list[i], lbl_dir, is_print=False)  # 复制文件
            if src_file_list[i].startswith(src_dir.rstrip('/') + '\\'):
                src_file_list[i] = src_file_list[i][len(src_dir.rstrip('/') + '\\'):]
        # breakpoint()  # 查看哪些被校正了, print correction_id

        # finetune
        train_dataset = CVC_ClinicDB_trnDataSet(img_dir, './dataset/corrected_masks/high0.5',
                                                './dataset/corrected_masks/high0.5/using_list.txt')
        trainloader = data.DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True)

        # model1, optimizer1, scheduler1 = setup_model(device, name='resnet101')
        model1.to(device)
        model1.cuda()
        optimizer1.param_groups[0]['lr'] = 0.0006
        optimizer1.param_groups[1]['lr'] = 0.00006
        optimizer1.param_groups[0]['momentum'] = 0.99
        optimizer1.param_groups[1]['momentum'] = 0.95
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=200, gamma=0.6)
        for epoch in range(20):
            print("  ******epoch=%d******" % epoch)
            print("train...,lr1 = %f, lr2 = %f" % (optimizer1.param_groups[0]['lr'], optimizer1.param_groups[1]['lr']))
            model1.train()
            pbar = tqdm(range(len(trainloader)))
            train_iter = iter(trainloader)
            for step_t in pbar:
                t_x, t_y, name = train_iter.next()
                t_x = t_x.float().to(device)
                t_y = t_y.long().to(device)
                if t_x.shape[0] < 2:
                    t_x = torch.vstack((t_x, t_x))
                    t_y = torch.vstack((t_y, t_y))
                optimizer1.zero_grad()
                outputs1 = model1(t_x)  # N,C,H,W
                loss_CrossEntropy = loss.myCrossEntropy2d(outputs1, t_y)
                DiceLoss = 20 * loss.myDiceLoss(outputs1, t_y, p=2, smooth=1e-8)
                Loss1 = loss_CrossEntropy + DiceLoss  # + focal_loss#+ Smoothloss
                Loss1.backward()
                optimizer1.step()
                scheduler1.step()
            model1.eval()
            print('testing')
            total_dice = 0
            tst_dataset = CVC_ClinicDB_tstDataSet(img_dir, lbl_clean_dir, test_list, means=train_dataset.means,
                                                  stdevs=train_dataset.stdevs)
            tstloader = data.DataLoader(tst_dataset, batch_size=1, num_workers=2, shuffle=False)
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
            # torch.save({
            #     "model_state": model1.module.state_dict(),
            #     "optimizer_state": optimizer1.state_dict(),
            #     "scheduler_state": scheduler1.state_dict(),
            # }, './dataset/it_correction_high0.5.pth')
            # print("model saved at %s" % './dataset/it_correction_high0.5.pth')


if __name__ == '__main__':
    del_file('./dataset/corrected_masks')
    pred_net = './dataset/High_noise/result0.5/clean_selected.pth'
    # breakpoint()
    label_corretion(pred_net)
