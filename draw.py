import os
import shutil
from glob import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F
import argparse
from PIL import Image
from tqdm import tqdm
from load_data import CVC_ClinicDB_noiseDataSet, CVC_ClinicDB_trnDataSet, CVC_ClinicDB_tstDataSet
import network
from network.utils import PolynomialLR
import loss
import matplotlib.pyplot as plt
import cv2
from train import setup_model

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

def predict():
    ##setup gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"
    flag = torch.cuda.is_available()
    print("GPU availabel:", flag)
    device = torch.device('cuda')
    print(torch.cuda.device_count())
    print("Device: %s" % device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())

    model1, _, _ = setup_model(device, 'resnet101')
    model1.to(device)
    model1.cuda()
    # pred_net = './dataset/result_gt/resnet101.pth'
    # pred_net = './dataset/High_noise/result0.5/resnet101.pth'
    # pred_net = './dataset/clean_selected.pth'
    pred_net = './dataset/low_noise/result0.3/resnet101.pth'
    checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
    model1.module.load_state_dict(checkpoint["model_state"])
    model1.to(device)
    model1.cuda()
    print("Model restored from %s" % pred_net)

    # test
    img_dir = './dataset/images/'
    lbl_dir = './dataset/using_masks/'
    lbl_clean_dir = './dataset/Ground Truth/'
    train_list = './dataset/train_list.txt'
    test_list = './dataset/test_list.txt'
    # img_dir = './dataset_K/images/'
    # lbl_dir = './dataset_K/using_masks/'
    # lbl_clean_dir = './dataset_K/masks/'
    # train_list = './dataset_K/train_list.txt'
    # test_list = './dataset_K/test_list.txt'

    train_dataset = CVC_ClinicDB_trnDataSet(img_dir, lbl_clean_dir, train_list)

    noise_dataset = CVC_ClinicDB_noiseDataSet(img_dir, lbl_dir, gt_root=lbl_clean_dir, list_path=train_list,
                                              means=train_dataset.means,
                                              stdevs=train_dataset.stdevs)
    # train_dataset = Kvasir_SEG_trnDataSet(img_dir, lbl_dir, train_list)
    # test_dataset = Kvasir_SEG_tstDataSet(img_dir, lbl_clean_dir, test_list, means=train_dataset.means,
    #                                        stdevs=train_dataset.stdevs)
    testloader = data.DataLoader(noise_dataset, batch_size=1)
    print("Test dataset:", len(testloader))
    total_dice = []
    model1.eval()
    pbar = tqdm(range(len(testloader)))
    test_iter = iter(testloader)
    img_ids = [i_id.strip() for i_id in open(train_list)]
    for step_t in pbar:
        t_x, t_y, gt, name = test_iter.next()
        # t_x = t_x.float().to(device)
        # outputs = model1(t_x)
        # outputs = nn.functional.softmax(outputs, dim=1)
        # preds = np.asarray(np.argmax(outputs.detach().cpu().numpy(), axis=1), dtype=np.uint8)
        # val_dice = loss.calculate_dice(preds[0], gt[0])
        # if val_dice > 1:
        #     breakpoint()
        # total_dice.append(val_dice)
        #breakpoint()
        # draw contours
        assert name[0] == img_ids[step_t]
        name = img_ids[step_t]
        img = cv2.imread(os.path.join(img_dir, name.replace('tif', 'jpg')), 3)  # bgr,h,w,3
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        # b, g, r = cv2.split(img)
        # img = cv2.merge([r, g, b])
        # img = np.asarray(img, np.float32)
        lbl = cv2.imread(os.path.join(lbl_dir, name),cv2.IMREAD_GRAYSCALE)  # 读取原始图片，彩色三通道图片
        lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_CUBIC)

        # if name == '69.tif':
        #     breakpoint()
        # if name == '577.tif':
        #     breakpoint()
        #
        # gray = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)
        # ret, binary = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY)  # 二值化灰度图图片，以127为门槛值，所有大于127像素值的像素取值255，所有小于127像素值的取值为0
        # contours_lbl, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓搜索，找到
        # _, contours_lbl, _ = cv2.findContours(255 * np.asarray(t_y[0], dtype='uint8'), cv2.RETR_TREE,
        #                                    cv2.CHAIN_APPROX_SIMPLE)
       # _,contours_pred,_ = cv2.findContours(255 * preds[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _,contours_gt,_ = cv2.findContours(255 * np.asarray(gt[0], dtype='uint8'), cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

       # breakpoint()
       # img = cv2.drawContours(img, contours_lbl, -1, (0, 0, 255), 2)  # 绘制轮廓 蓝色为gt,红色为预测
       # img = cv2.drawContours(img, contours_pred, -1, (0, 0, 255), 2)  # 绘制轮廓 红色预测
        img = cv2.drawContours(img, contours_gt, -1, (0, 255, 0), 2)  # 绘制轮廓 蓝为gt
        # font = cv2.FONT_HERSHEY_SIMPL EX
        # img = cv2.putText(img, str(val_dice)[0:6], (0, 40), font, 1.0, (255, 255, 255), 2)  # 添加文字，
        #img =255* preds[0]
        # if not cv2.imwrite(os.path.join('dataset/draw/', name.replace('.tif', '.jpg')), img):
        #     # if not cv2.imwrite(os.path.join('./dataset/High_noise/result0.5', name.replace('.tif', '.jpg')), img):
        #     breakpoint()
        #
        # heatmapshow = None
        # #breakpoint()
        # heatmapshow = cv2.normalize(lbl, heatmapshow, alpha=0, beta=255,
        #                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # heatmapshow = cv2.applyColorMap(heatmapshow, 2)
        # if not cv2.imwrite(os.path.join('dataset/draw/lbl/', name.replace('.tif', '.jpg')),
        #                    heatmapshow):
        #     # if not cv2.imwrite(os.path.join('./dataset/High_noise/result0.5', name.replace('.tif', '.jpg')), img):
        #     breakpoint()

        heatmapshow = None
        #breakpoint()
        # plt.figure(figsize=(2.56, 2.56))
        # plt.imshow(gt[0])
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        plt.imsave(os.path.join('dataset/draw/lbl/', name.replace('.tif', '.jpg')), lbl)
        plt.imsave(os.path.join('dataset/draw/gt/', name.replace('.tif', '.jpg')), gt[0])
        # heatmapshow = None
        # gt = cv2.imread(os.path.join(lbl_clean_dir, name))  # 读取原始图片，彩色三通道图片
        # gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_CUBIC)
        # heatmapshow = cv2.normalize(gt, heatmapshow, alpha=0, beta=255,
        #                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # heatmapshow = cv2.applyColorMap(heatmapshow, 2)
        # if not cv2.imwrite(os.path.join('dataset/draw/gt/', name.replace('.tif', '.jpg')),
        #                    heatmapshow):
        #     # if not cv2.imwrite(os.path.join('./dataset/High_noise/result0.5', name.replace('.tif', '.jpg')), img):
        #     breakpoint()


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
    noise_dir = './dataset/Low_noise/rate0.5/'
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
    predict()