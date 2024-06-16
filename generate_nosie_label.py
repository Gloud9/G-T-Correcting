import sys

import torch
from torchvision import transforms
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
import time
import cv2, os
import numpy as np
import scipy
from skimage.measure import label, regionprops
from skimage import data,color,morphology,feature,measure,draw
import scipy.ndimage as ndimage
from PIL import Image
import matplotlib.pyplot as plt
import os
import Augmentor
# import elasticdeform
from skimage import measure
#from skimage.measure import  measure.compare_psnr
import random


def get_bbox(img):
    h, w = img.shape
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    r = rmax - rmin
    c = cmax - cmin
    x = np.round((rmax+rmin)/2)
    y = np.round((cmax+cmin)/2)
    x1 = x-256
    x2 = x+256
    y1 = y-256
    y2 = y+256
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    return np.uint16(y1), np.uint16(x1), np.uint16(y2), np.uint16(x2)


def get_bbox_source(img):
    h, w = img.shape
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    r = rmax - rmin
    c = cmax - cmin
    x = np.round((rmax+rmin)/2)
    y = np.round((cmax+cmin)/2)
    x1 = x-256
    x2 = x+256
    y1 = y-256
    y2 = y+256
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    return np.uint16(y1), np.uint16(x1), np.uint16(y2), np.uint16(x2)

def get_bool(img, type = 0):
    if type == 0:
        return img == 0
    if type == 1:
        return img == 1

def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)

def random_erosion(source_root, root, filelist,flag=1, magnitude=(20,20)):
    #random binary erosion
    noise_ratio = []
    for i in range(len(filelist)):
        file = filelist[i]
        #print(i, file)
        gt_mask = Image.open(os.path.join(source_root, file))
        gt_mask = gt_mask.convert('1')
        gt_mask = np.asarray(gt_mask, np.uint8)
        id_mask = np.asarray(gt_mask, np.float32)
        mask_to_id = {0: 0, 255: 1}
        for k, v in mask_to_id.items():
            id_mask[gt_mask == k] = v
        mask = get_bool(id_mask, 1)
        if flag == 1:
            mask_binary_erosion = morphology.binary_erosion(mask, morphology.disk(magnitude[0])).astype(np.uint8)
            output_mask = mask_binary_erosion
            output_mask = output_mask * 255
        if flag == 2:
            mask_binary_erosion = morphology.binary_dilation(mask, morphology.disk(magnitude[1])).astype(np.uint8)
            output_mask = mask_binary_erosion
            output_mask = output_mask * 255

        if len(np.unique(output_mask)) > 2:
            sys.exit(0)
        if len(np.unique(id_mask)) > 2:
            sys.exit(0)
        # plt.figure(figsize=(10, 10))
        # plt.subplot(121)
        # plt.imshow(255*id_mask, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(output_mask, cmap='gray')
        # plt.show()
        #print(dice_coef(id_mask, output_mask/255))

        noise = 1-dice_coef(id_mask, output_mask/255)
        noise_ratio.append(noise)
        #print('noise degree:', noise)
        output_mask = Image.fromarray(output_mask.astype(np.uint8))
        output_mask.save(os.path.join(root, file))


    print("average noise rate: ", np.mean(noise_ratio))
    return np.mean(noise_ratio)

def random_distortion(source_root, save_root, filelist, flag=1):

    p = Augmentor.Pipeline()
    if flag == 1:
        p.random_distortion(probability=1, grid_height=30, grid_width=30, magnitude=10)
    if flag == 2:
        p.random_distortion(probability=1, grid_height=30, grid_width=30, magnitude=10)
    distort_trans = transforms.Compose([
        p.torch_transform(),
    ])
    noise_ratio = []
    for i in range(len(filelist)):
        file = filelist[i]
        #print(i, file)
        gt_mask = Image.open(os.path.join(source_root, file))
        output_mask = distort_trans(gt_mask)
        output_mask = output_mask.convert("1")
        output_mask = np.asarray(output_mask, np.uint8)
        gt_mask = gt_mask.convert("1")
        id_mask = np.asarray(gt_mask, np.uint8)

        output_mask = output_mask*255

        if len(np.unique(output_mask)) > 2:
            sys.exit(0)
        if len(np.unique(id_mask)) > 2:
            sys.exit(0)
        # plt.figure(figsize=(10, 10))
        # plt.subplot(121)
        # plt.imshow(255*id_mask, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(output_mask, cmap='gray')
        # plt.show()
        #print(dice_coef(id_mask, output_mask/255))

        noise = 1-dice_coef(id_mask, output_mask/255)
        noise_ratio.append(noise)
        #print('noise degree:', noise)
        output_mask = Image.fromarray(output_mask.astype(np.uint8))
        output_mask.save(os.path.join(save_root, file))
    print("average noise rate: ", np.mean(noise_ratio))
    return np.mean(noise_ratio)

def random_aff(source_root, save_root, filelist, translate=(0.05, 0.05)):
    aff_trans = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=translate),
    ])
    noise_ratio = []
    for i in range(len(filelist)):
        file = filelist[i]
        #print(i, file)
        gt_mask = Image.open(os.path.join(source_root, file))
        output_mask = aff_trans(gt_mask)
        output_mask = output_mask.convert("1")
        output_mask = np.asarray(output_mask, np.uint8)
        gt_mask = gt_mask.convert("1")
        id_mask = np.asarray(gt_mask, np.uint8)

        id_mask = id_mask * 255
        output_mask = output_mask * 255

        if len(np.unique(output_mask)) > 2:
            sys.exit(0)
        if len(np.unique(id_mask)) > 2:
            sys.exit(0)
        # plt.figure(figsize=(10, 10))
        # plt.subplot(121)
        # plt.imshow(id_mask, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(output_mask, cmap='gray')
        # plt.show()
        # print(dice_coef(id_mask/255, output_mask/255))

        noise = 1-dice_coef(id_mask/255, output_mask/255)
        noise_ratio.append(noise)
        #print('noise degree:', noise)
        output_mask = Image.fromarray(output_mask.astype(np.uint8))
        output_mask.save(os.path.join(save_root, file))
    print("average noise rate: ", np.mean(noise_ratio))
    return np.mean(noise_ratio)

def generate_low_noise(gt_root, save_root, select_img):
    magnitude = (10, 15)
    print("\n************generate_low_noise***************")
    print("random erosion")
    random_erosion(gt_root, save_root, select_img[0:int(0.33 * len(select_img))], flag=1, magnitude=magnitude)
    print("random dilation")
    random_erosion(gt_root, save_root, select_img[int(0.33*len(select_img)):int(0.66*len(select_img))], flag=2, magnitude=magnitude)
    # affine transform
    print("random affine transformation")
    random_aff(gt_root, save_root, select_img[int(0.66 * len(select_img)):len(select_img)], translate=(0.07, 0.07))
    distort_img = random.sample(select_img, int(0.5 * len(select_img)))
  #  random_distortion(gt_root, save_root, distort_img)
def generate_verylow_noise(gt_root, save_root, select_img):
    magnitude = (5, 10)
    print("\n************generate_low_noise***************")
    print("random erosion")
    random_erosion(gt_root, save_root, select_img[0:int(0.33 * len(select_img))], flag=1, magnitude=(5, 5))
    print("random dilation")
    random_erosion(gt_root, save_root, select_img[int(0.33*len(select_img)):int(0.66*len(select_img))], flag=2, magnitude=(3, 3))
    # affine transform
    print("random affine transformation")
    random_aff(gt_root, save_root, select_img[int(0.66 * len(select_img)):len(select_img)], translate=(0.03, 0.03))
    distort_img = random.sample(select_img, int(0.5 * len(select_img)))
  #  random_distortion(gt_root, save_root, distort_img)



def generate_high_noise(gt_root, save_root, select_img):
    magnitude = (18, 28)
    print("\n************generate_high_noise***************")
    print("random erosion")
    random_erosion(gt_root, save_root, select_img[0:int(0.33 * len(select_img))], flag=1, magnitude=magnitude)
    print("random dilation")
    random_erosion(gt_root, save_root, select_img[int(0.33*len(select_img)):int(0.66*len(select_img))], flag=2, magnitude=magnitude)
    # affine transform
    print("random affine transformation")
    random_aff(gt_root, save_root, select_img[int(0.66 * len(select_img)):len(select_img)], translate=(0.12, 0.12))
    print("random distortion")
    random.sample(select_img, int(0.5*len(select_img)))
    #average_noise_rate4 = random_distortion(gt_root, save_root, distort_img)



def main():

    img_root = './dataset/Original'
    gt_root = './dataset/Ground Truth'
    Hnoise_root = './dataset/High_noise'
    Lnoise_root = './dataset/Low_noise'
    Lnoise_root2 = './dataset/Less_noise'
    train_list = './dataset/train_list.txt'
    def del_file(path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                del_file(c_path)
            else:
                os.remove(c_path)

    # if not os.path.exists(Hnoise_root):
    #     os.makedirs(Hnoise_root)
    # if not os.path.exists(Lnoise_root):
    #     os.makedirs(Lnoise_root)
    # del_file(Hnoise_root)
    # del_file(Lnoise_root)
    # if not os.path.exists(os.path.join(Lnoise_root, 'rate0.1')):
    #     os.makedirs(os.path.join(Lnoise_root, 'rate0.1'))
    # if not os.path.exists(os.path.join(Lnoise_root, 'rate0.5')):
    #     os.makedirs(os.path.join(Lnoise_root, 'rate0.5'))
    # if not os.path.exists(os.path.join(Lnoise_root, 'rate0.9')):
    #     os.makedirs(os.path.join(Lnoise_root, 'rate0.9'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'rate0.1')):
    #     os.makedirs(os.path.join(Hnoise_root, 'rate0.1'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'rate0.5')):
    #     os.makedirs(os.path.join(Hnoise_root, 'rate0.5'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'rate0.9')):
    #     os.makedirs(os.path.join(Hnoise_root, 'rate0.9'))
    # if not os.path.exists(os.path.join(Lnoise_root, 'result0.1')):
    #     os.makedirs(os.path.join(Lnoise_root, 'result0.1'))
    # if not os.path.exists(os.path.join(Lnoise_root, 'result0.5')):
    #     os.makedirs(os.path.join(Lnoise_root, 'result0.5'))
    # if not os.path.exists(os.path.join(Lnoise_root, 'result0.9')):
    #     os.makedirs(os.path.join(Lnoise_root, 'result0.9'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'result0.1')):
    #     os.makedirs(os.path.join(Hnoise_root, 'result0.1'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'result0.5')):
    #     os.makedirs(os.path.join(Hnoise_root, 'result0.5'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'result0.9')):
    #     os.makedirs(os.path.join(Hnoise_root, 'result0.9'))
    if not os.path.exists(os.path.join(Lnoise_root2, 'result0.3')):
        os.makedirs(os.path.join(Lnoise_root2, 'result0.3'))
        if not os.path.exists(os.path.join(Lnoise_root2, 'result0.1')):
            os.makedirs(os.path.join(Lnoise_root2, 'result0.1'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'result0.3')):
    #     os.makedirs(os.path.join(Hnoise_root, 'result0.3'))
    if not os.path.exists(os.path.join(Lnoise_root2, 'result0.5')):
        os.makedirs(os.path.join(Lnoise_root2, 'result0.5'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'result0.7')):
    #     os.makedirs(os.path.join(Hnoise_root, 'result0.7'))
    if not os.path.exists(os.path.join(Lnoise_root2, 'rate0.3')):
        os.makedirs(os.path.join(Lnoise_root2, 'rate0.3'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'rate0.3')):
    #     os.makedirs(os.path.join(Hnoise_root, 'rate0.3'))
    if not os.path.exists(os.path.join(Lnoise_root2, 'rate0.5')):
        os.makedirs(os.path.join(Lnoise_root2, 'rate0.5'))
    if not os.path.exists(os.path.join(Lnoise_root2, 'rate0.1')):
        os.makedirs(os.path.join(Lnoise_root2, 'rate0.1'))
    # if not os.path.exists(os.path.join(Hnoise_root, 'rate0.7')):
    #     os.makedirs(os.path.join(Hnoise_root, 'rate0.7'))
    # H3_root = os.path.join(Hnoise_root, 'rate0.3')
    # L3_root = os.path.join(Lnoise_root, 'rate0.3')
    # H7_root = os.path.join(Hnoise_root, 'rate0.7')
    # L7_root = os.path.join(Lnoise_root, 'rate0.7')

    L1_root2 = os.path.join(Lnoise_root2, 'rate0.1')
    L3_root2 = os.path.join(Lnoise_root2, 'rate0.3')
    L5_root2 = os.path.join(Lnoise_root2, 'rate0.5')
    # H1_root = os.path.join(Hnoise_root, 'rate0.1')
    # H5_root = os.path.join(Hnoise_root, 'rate0.5')
    # H9_root = os.path.join(Hnoise_root, 'rate0.9')
    # L1_root = os.path.join(Lnoise_root, 'rate0.1')
    # L5_root = os.path.join(Lnoise_root, 'rate0.5')
    # L9_root = os.path.join(Lnoise_root, 'rate0.9')
    torch.manual_seed(1024)
    ##高噪声扩张收缩在0.5-0.7DICE，低噪声在0.2-0.3
    ##高噪声变形在0.08DICE,低噪声在0.04
    # 更低噪声生成
    breakpoint()
    imglist = [i_id.strip() for i_id in open(train_list)]
    #imglist = os.listdir(gt_root)
    # 收缩，扩张，弹性扭曲，腐蚀 10%,40%,70%,100% low,mid,high
    # select_img = random.sample(imglist, int(0.1 * len(imglist)))
    # generate_low_noise(gt_root, L1_root, select_img)
    # select_img = random.sample(imglist, int(0.5 * len(imglist)))
    # generate_low_noise(gt_root, L5_root, select_img)
    # select_img = random.sample(imglist, int(0.9 * len(imglist)))
    # generate_low_noise(gt_root, L9_root, select_img)
    select_img = random.sample(imglist, int(0.1 * len(imglist)))
    generate_verylow_noise(gt_root, L1_root2, select_img)
    select_img = random.sample(imglist, int(0.3 * len(imglist)))
    generate_verylow_noise(gt_root, L3_root2, select_img)
    select_img = random.sample(imglist, int(0.5 * len(imglist)))
    generate_verylow_noise(gt_root, L5_root2, select_img)
    # 低噪声生成
    breakpoint()
    imglist = [i_id.strip() for i_id in open(train_list)]
    #imglist = os.listdir(gt_root)
    # 收缩，扩张，弹性扭曲，腐蚀 10%,40%,70%,100% low,mid,high
    # select_img = random.sample(imglist, int(0.1 * len(imglist)))
    # generate_low_noise(gt_root, L1_root, select_img)
    # select_img = random.sample(imglist, int(0.5 * len(imglist)))
    # generate_low_noise(gt_root, L5_root, select_img)
    # select_img = random.sample(imglist, int(0.9 * len(imglist)))
    # generate_low_noise(gt_root, L9_root, select_img)
    select_img = random.sample(imglist, int(0.3 * len(imglist)))
    generate_low_noise(gt_root, L3_root, select_img)
    select_img = random.sample(imglist, int(0.7 * len(imglist)))
    generate_low_noise(gt_root, L7_root, select_img)
    breakpoint()
    #高噪声生成
    imglist = [i_id.strip() for i_id in open(train_list)]
   # 收缩，扩张，弹性扭曲，腐蚀 10%,40%,70%,100% low,mid,high
   #  select_img = random.sample(imglist, int(0.1*len(imglist)))
   #  generate_high_noise(gt_root, H1_root, select_img)
   #  select_img = random.sample(imglist, int(0.5*len(imglist)))
   #  generate_high_noise(gt_root, H5_root, select_img)
   #  select_img = random.sample(imglist, int(0.9*len(imglist)))
   #  generate_high_noise(gt_root, H9_root, select_img)
    select_img = random.sample(imglist, int(0.3*len(imglist)))
    generate_high_noise(gt_root, H3_root, select_img)
    select_img = random.sample(imglist, int(0.7*len(imglist)))
    generate_high_noise(gt_root, H7_root, select_img)

if __name__ == '__main__':
    main()
