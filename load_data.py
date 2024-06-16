import os
import os.path as osp
import sys

import numpy as np
import random
import collections
import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler
import itertools


class CVC_ClinicDB_trnDataSet(data.Dataset):
    def __init__(self, img_root, label_root, list_path, means=None, stdevs=None, crop_size=None,):
        if crop_size is None:
            crop_size = (256, 256)
        self.img_root = img_root
        self.label_root = label_root
        self.list_path = list_path
        file = open(list_path)
        self.img_ids = [i_id.strip() for i_id in file]
        file.close()
        self.colormap = {0: 0, 255: 1}
        self.crop_size = crop_size
        if np.array(means == None).any():
            self.means = np.zeros(3) #[-44.04378088 -34.9654579   19.20693703]
            self.stdevs = np.zeros(3) #[44.80745859 54.39029705 77.75020803]
            self.get_mean_std()
        else:
            self.means = means
            self.stdevs = stdevs



    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        #print(name)
        img = Image.open(os.path.join(self.img_root, name.replace('.tif', '.jpg')))
        img = img.convert("RGB")
        img = img.resize(self.crop_size, Image.BICUBIC)
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # img: (h,w,3) BGR
        img -= np.array((104, 117, 123))  # BGR
        img = img.transpose((2, 0, 1))  # img:(3,h,w)
        lbl = Image.open(os.path.join(self.label_root, name))
        lbl = lbl.convert('1')
        lbl = lbl.resize(self.crop_size, Image.BICUBIC)
        lbl = np.asarray(lbl, np.float32)
        lbl = lbl*255
        # breakpoint()
        if len(np.unique(lbl)) > 2:
            sys.exit(0)

        for k, v in self.colormap.items():
            lbl[lbl == k] = v
        img[0] = (img[0] - self.means[0])/self.stdevs[0]
        img[1] = (img[1] - self.means[1]) / self.stdevs[1]
        img[2] = (img[2] - self.means[2]) / self.stdevs[2]
        return img.copy(), lbl.copy(), name

    def get_mean_std(self):
        """
        计算数据集的均值和标准差
        :param type: 使用的是那个数据集的数据，有'train', 'test', 'testing'
        :param mean_std_path: 计算出来的均值和标准差存储的文件
        :return:
        """
        num_imgs = len(self.img_ids)
        for file in self.img_ids:
            img = Image.open(os.path.join(self.img_root, file.replace('.tif', '.jpg')))
            img = img.convert("RGB")
            img = img.resize(self.crop_size, Image.BICUBIC)
            img = np.asarray(img, np.float32)

            img = img[:, :, ::-1]  # img: (h,w,3) BGR
            img -= np.array((104, 117, 123))  # BGR
            img = img.transpose((2, 0, 1))  # img:(3,h,w)
            for i in range(3):
                # 一个通道的均值和标准差
                self.means[i] += img[i, :, : ].mean()
                self.stdevs[i] += img[i, :, : ].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs

        print("{} : normMean = {}".format(type, self.means))
        print("{} : normstdevs = {}".format(type, self.stdevs))



class CVC_ClinicDB_tstDataSet(data.Dataset):
    def __init__(self, img_root, label_root, list_path, means,stdevs,crop_size=(256, 256)):
        if crop_size is None:
            crop_size = (256, 256)
        self.img_root = img_root
        self.label_root = label_root
        self.list_path = list_path
        file = open(list_path)
        self.img_ids = [i_id.strip() for i_id in file]
        file.close()
        self.colormap = {0: 0, 255: 1}
        self.crop_size = crop_size
        self.means = means #[-44.04378088 -34.9654579   19.20693703]
        self.stdevs = stdevs #[44.80745859 54.39029705 77.75020803]



    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        #print(name)
        img = Image.open(os.path.join(self.img_root, name.replace('.tif', '.jpg')))
        img = img.convert("RGB")
        img = img.resize(self.crop_size, Image.BICUBIC)
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # img: (h,w,3) BGR
        img -= np.array((104, 117, 123))  # BGR
        img = img.transpose((2, 0, 1))  # img:(3,h,w)
        lbl = Image.open(os.path.join(self.label_root, name))
        lbl = lbl.convert('1')
        lbl = lbl.resize(self.crop_size, Image.BICUBIC)
        lbl = np.asarray(lbl, np.float32)
        lbl = lbl*255
        # breakpoint()
        if len(np.unique(lbl)) > 2:
            sys.exit(0)

        for k, v in self.colormap.items():
            lbl[lbl == k] = v
        img[0] = (img[0] - self.means[0])/self.stdevs[0]
        img[1] = (img[1] - self.means[1]) / self.stdevs[1]
        img[2] = (img[2] - self.means[2]) / self.stdevs[2]
        return img.copy(), lbl.copy(), name

class CVC_ClinicDB_noiseDataSet(data.Dataset):
    def __init__(self, img_root, noise_label_root,gt_root, list_path, means,stdevs,crop_size=(256, 256)):
        if crop_size is None:
            crop_size = (256, 256)
        self.img_root = img_root
        self.noise_label_root = noise_label_root
        self.gt_root = gt_root
        self.list_path = list_path
        file = open(list_path)
        self.img_ids = [i_id.strip() for i_id in file]
        file.close()
        self.colormap = {0: 0, 255: 1}
        self.crop_size = crop_size
        self.means = means #[-44.04378088 -34.9654579   19.20693703]
        self.stdevs = stdevs #[44.80745859 54.39029705 77.75020803]



    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        #print(name)
        img = Image.open(os.path.join(self.img_root, name.replace('.tif', '.jpg')))
        img = img.convert("RGB")
        img = img.resize(self.crop_size, Image.BICUBIC)
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # img: (h,w,3) BGR
        img -= np.array((104, 117, 123))  # BGR
        img = img.transpose((2, 0, 1))  # img:(3,h,w)
        lbl = Image.open(os.path.join(self.noise_label_root, name))
        lbl = lbl.convert('1')
        lbl = lbl.resize(self.crop_size, Image.BICUBIC)
        lbl = np.asarray(lbl, np.float32)
        lbl = lbl*255

        gt = Image.open(os.path.join(self.gt_root, name))
        gt = gt.convert('1')
        gt = gt.resize(self.crop_size, Image.BICUBIC)
        gt = np.asarray(gt, np.float32)
        gt = gt*255
        # breakpoint()
        if len(np.unique(lbl)) > 2:
            sys.exit(0)

        for k, v in self.colormap.items():
            lbl[lbl == k] = v
        for k, v in self.colormap.items():
            gt[gt == k] = v
        img[0] = (img[0] - self.means[0])/self.stdevs[0]
        img[1] = (img[1] - self.means[1]) / self.stdevs[1]
        img[2] = (img[2] - self.means[2]) / self.stdevs[2]
        return img.copy(), lbl.copy(), gt.copy(), name


class Kvasir_SEG_trnDataSet(data.Dataset):
    def __init__(self, img_root, label_root, list_path, crop_size=None):
        if crop_size is None:
            crop_size = (256, 256)
        self.img_root = img_root
        self.label_root = label_root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.colormap = {0: 0, 255: 1}
        self.crop_size = crop_size
        self.means = np.zeros(3) #[-44.04378088 -34.9654579   19.20693703]
        self.stdevs = np.zeros(3) #[44.80745859 54.39029705 77.75020803]
        self.get_mean_std()



    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        #print(name)
        img = Image.open(os.path.join(self.img_root, name))
        img = img.convert("RGB")
        img = img.resize(self.crop_size, Image.BICUBIC)
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # img: (h,w,3) BGR
        img -= np.array((104, 117, 123))  # BGR
        img = img.transpose((2, 0, 1))  # img:(3,h,w)
        lbl = Image.open(os.path.join(self.label_root, name))
        lbl = lbl.convert('1')
        lbl = lbl.resize(self.crop_size, Image.BICUBIC)
        lbl = np.asarray(lbl, np.float32)
        lbl = lbl*255
        # breakpoint()
        if len(np.unique(lbl)) > 2:
            sys.exit(0)

        for k, v in self.colormap.items():
            lbl[lbl == k] = v
        img[0] = (img[0] - self.means[0])/self.stdevs[0]
        img[1] = (img[1] - self.means[1]) / self.stdevs[1]
        img[2] = (img[2] - self.means[2]) / self.stdevs[2]
        return img.copy(), lbl.copy(), name

    def get_mean_std(self):
        """
        计算数据集的均值和标准差
        :param type: 使用的是那个数据集的数据，有'train', 'test', 'testing'
        :param mean_std_path: 计算出来的均值和标准差存储的文件
        :return:
        """
        num_imgs = len(self.img_ids)
        for file in self.img_ids:
            img = Image.open(os.path.join(self.img_root, file))
            img = img.convert("RGB")
            img = img.resize(self.crop_size, Image.BICUBIC)
            img = np.asarray(img, np.float32)

            img = img[:, :, ::-1]  # img: (h,w,3) BGR
            img -= np.array((104, 117, 123))  # BGR
            img = img.transpose((2, 0, 1))  # img:(3,h,w)
            for i in range(3):
                # 一个通道的均值和标准差
                self.means[i] += img[i, :, : ].mean()
                self.stdevs[i] += img[i, :, : ].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs

        print("{} : normMean = {}".format(type, self.means))
        print("{} : normstdevs = {}".format(type, self.stdevs))

class Kvasir_SEG_tstDataSet(data.Dataset):
    def __init__(self, img_root, label_root, list_path, means,stdevs,crop_size=(256, 256)):
        if crop_size is None:
            crop_size = (256, 256)
        self.img_root = img_root
        self.label_root = label_root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.colormap = {0: 0, 255: 1}
        self.crop_size = crop_size
        self.means = means #[-44.04378088 -34.9654579   19.20693703]
        self.stdevs = stdevs #[44.80745859 54.39029705 77.75020803]



    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        #print(name)
        img = Image.open(os.path.join(self.img_root, name))
        img = img.convert("RGB")
        img = img.resize(self.crop_size, Image.BICUBIC)
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # img: (h,w,3) BGR
        img -= np.array((104, 117, 123))  # BGR
        img = img.transpose((2, 0, 1))  # img:(3,h,w)
        lbl = Image.open(os.path.join(self.label_root, name))
        lbl = lbl.convert('1')
        lbl = lbl.resize(self.crop_size, Image.BICUBIC)
        lbl = np.asarray(lbl, np.float32)
        lbl = lbl*255
        # breakpoint()
        if len(np.unique(lbl)) > 2:
            sys.exit(0)

        for k, v in self.colormap.items():
            lbl[lbl == k] = v
        img[0] = (img[0] - self.means[0])/self.stdevs[0]
        img[1] = (img[1] - self.means[1]) / self.stdevs[1]
        img[2] = (img[2] - self.means[2]) / self.stdevs[2]
        return img.copy(), lbl.copy(), name

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)