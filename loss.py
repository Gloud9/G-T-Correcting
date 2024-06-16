import torch.nn as nn
import torch.nn.functional as F
import torch 
from torch.autograd import Variable
from skimage.measure import label, regionprops
import scipy.ndimage as ndimage
import numpy as np
from PIL import Image
import os

def Smoothloss( predict, target):
    # import pdb;pdb.set_trace()
    assert not target.requires_grad
    n, c, h, w = predict.size()
    predict = torch.sigmoid(predict)

    loss = abs(predict[:, 0, 1: h - 1, 1: w - 1] - predict[:, 0, 0: h - 2, 1: w - 1]) + \
           abs(predict[:, 0, 1: h - 1, 1: w - 1] - predict[:, 0, 2: h, 1: w - 1]) + \
           abs(predict[:, 0, 1: h - 1, 1: w - 1] - predict[:, 0, 1: h - 1, 0: w - 2]) + \
           abs(predict[:, 0, 1: h - 1, 1: w - 1] - predict[:, 0, 1: h - 1, 2: w])

    M1 = torch.zeros(loss.shape).cuda()
    M2 = torch.zeros(loss.shape).cuda()
    M3 = torch.zeros(loss.shape).cuda()
    M4 = torch.zeros(loss.shape).cuda()

    M1[target[:, 1: h - 1, 1: w - 1] == target[:, 0: h - 2, 1: w - 1]] = 1
    M2[target[:, 1: h - 1, 1: w - 1] == target[:, 2: h, 1: w - 1]] = 1
    M3[target[:, 1: h - 1, 1: w - 1] == target[:, 1: h - 1, 0: w - 2]] = 1
    M4[target[:, 1: h - 1, 1: w - 1] == target[:, 1: h - 1, 2: w]] = 1
    loss = loss * M1 * M2 * M3 * M4
    loss = loss.mean()
    return loss

def myCrossEntropy2d(predict, target):
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
    assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
    assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
    n, c, h, w = predict.size()
    target_mask = (target >= 0)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    # predict= F.softmax(predict, dim=1)
    # log_soft_out = torch.log(predict)
    # loss = F.nll_loss(log_soft_out, target)
    loss = F.cross_entropy(predict, target, weight=None, reduction='mean')
    return loss

# def reweight_CrossEntropy2d(predict, target):
#     #assert not target.requires_grad
#     assert predict.dim() == 4
#    # assert target.dim() == 3
#    #  assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
#    #  assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
#    #  assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
#
#
#     n, c, h, w = predict.size()
#     weight_map = torch.zeros((n, c, h, w))
#
#     target=target.detach().cpu()
#
#     for i in range(n):
#
#         mask = caculate_weight_map(target[i] * 255)
#         weight_map[i]= torch.from_numpy(mask)
#
#     weight_map=weight_map.cuda().long()
#
#
#     target_mask = (target >= 0)
#     # if not target.data.dim():
#     #     return Variable(torch.zeros(1))
#     predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
#     predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
#     weight_map = weight_map.transpose(1, 2).transpose(2, 3).contiguous()
#     weight_map = weight_map[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
#     predict = F.softmax(predict, dim=1)
#     log_soft_out = torch.log(predict)
#
#     log_soft_out1 = log_soft_out[:,0].view(-1,1)
#     weight_map1 = weight_map[:,0]
#     log_soft_out2 = log_soft_out[:,1].view(-1,1)
#     weight_map2 = weight_map[:,1]
#     log_soft_out3 = log_soft_out[:,2].view(-1,1)
#     weight_map3 = weight_map[:,2]
#
#     loss = F.nll_loss(log_soft_out1, weight_map1)
#     loss += F.nll_loss(log_soft_out2, weight_map2)
#     loss += F.nll_loss(log_soft_out3, weight_map3)
#     return loss

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    #import pdb;pdb.set_trace()
    #print('one-hot',input.size())
    input = input.cpu()
    input = np.array(input)
    shape = np.array(input.shape)
    #shape = np.expand_dims(shape, axis=1)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    input = Variable(torch.from_numpy(input)).type(torch.LongTensor)
    result = result.scatter_(1, input, 1)
    result = Variable(result[:, :3, :, :]).cuda()
    return result



def BinaryDiceLoss(predict, target,p=1,smooth=1):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    #import p#db;pdb.set_trace()
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = torch.sum(torch.mul(predict, target))*2 + smooth
    den = torch.sum(predict.pow(p) + target.pow(p)) + smooth

    dice = num / den
    loss = 1 - dice
    return loss


def myDiceLoss(predict, target,p=1,smooth=1):
    """Dice loss, need one hot encode input
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of shape [N, *]
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    #import pdb;pdb.set_trace()
    assert not target.requires_grad
    target = target.view((target.shape[0], 1, *target.shape[1:]))
    target = make_one_hot(target, 2)
    assert predict.shape == target.shape, 'predict & target shape do not match'

    total_loss = 0
    predict = F.softmax(predict, dim=1)
    #import pdb;pdb.set_trace()
    for i in range(target.shape[1]):
        dice_loss = BinaryDiceLoss(predict[:, i], target[:, i],p,smooth)
        total_loss += dice_loss
    loss = total_loss/target.shape[1]
    return loss

def my_weighted_DiceLoss(predict, target, batch_weight, p=1,smooth=1):
    """Dice loss, need one hot encode input
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of shape [N, *]
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    #import pdb;pdb.set_trace()
    assert not target.requires_grad
    target = target.view((target.shape[0], 1, *target.shape[1:]))
    target = make_one_hot(target, 2)
    assert predict.shape == target.shape, 'predict & target shape do not match'

    total_loss = 0
    predict = F.softmax(predict, dim=1)
    #import pdb;pdb.set_trace()
    for i in range(target.shape[1]):
        dice_loss = BinaryDiceLoss(predict[:, i], target[:, i],p,smooth)
        total_loss += dice_loss
    loss = total_loss/target.shape[1]
    return loss
# def reweight_DiceLoss(predict, target,p=1,smooth=1):
#     """Dice loss, need one hot encode input
#     Args:
#         predict: A tensor of shape [N, C, *]
#         target: A tensor of shape [N, *]
#         other args pass to BinaryDiceLoss
#     Return:
#         same as BinaryDiceLoss
#     """
#     #import pdb;pdb.set_trace()
#
#
#     total_loss = 0
#
#
#     n, c, h, w = predict.size()
#
#
#     weight_map = torch.zeros((n, c, h, w))
#
#     target=target.detach().cpu()
#
#     for i in range(n):
#
#         mask = caculate_weight_map(target[i] * 255)
#         weight_map[i]= torch.from_numpy(mask)
#
#     weight_map=weight_map.cuda().long()
#
#     predict = F.softmax(predict, dim=1)
#     assert predict.shape == weight_map.shape, 'predict & target shape do not match'
#     #import pdb;pdb.set_trace()
#     for i in range(weight_map.shape[1]):
#         dice_loss = BinaryDiceLoss(predict[:, i], weight_map[:, i],p,smooth)
#         total_loss += dice_loss
#     loss = total_loss/weight_map.shape[1]
#     return loss

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    #return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

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

def calculate_mIOU(pred_mask, true_mask):
    """mIOU
    Args:
        predict: A tensor of shape [H, W]
        target: A tensor of shape [H, W]
    Return:
        mIOU
    """
    H=true_mask.shape[0]
    W=true_mask.shape[1]
    pred = np.zeros((H, W))
    true = np.zeros((H, W))
    # disc_mIOU
    true[true_mask == 0] = 1
    true[true_mask == 1] = 1
    true[true_mask == 2] = 0
    pred[pred_mask == 0] = 1
    pred[pred_mask == 1] = 1
    pred[pred_mask == 2] = 0
    TP = sum(sum(true * pred))
    FP = sum(sum((pred == 1) & (true == 0)))
    FN = sum(sum((pred == 0) & (true == 1)))
    disc_mIOU = TP/(FP+FN+TP)
    # CUP_mIOU
    true[true_mask == 0] = 1
    true[true_mask == 1] = 0
    true[true_mask == 2] = 0
    pred[pred_mask == 0] = 1
    pred[pred_mask == 1] = 0
    pred[pred_mask == 2] = 0
    TP = sum(sum(true * pred))
    FP = sum(sum((pred == 1) & (true == 0)))
    FN = sum(sum((pred == 0) & (true == 1)))
    cup_mIOU = TP/(FP+FN+TP)

    return disc_mIOU, cup_mIOU

# def caculate_weight_map(mask, weight_cof=30):
#    # mask = mask.resize((1024, 1024), Image.NEAREST)
#     mask = np.asarray(mask, np.float32)
#     mask_cup = np.zeros(mask.shape, dtype=np.float32)
#     ind = {0: 255, 128: 0, 255: 0}
#     for k, v in ind.items():
#         mask_cup[mask == k] = v
#
#     mask_disc = np.zeros(mask.shape, dtype=np.float32)
#     ind = {0: 255, 128: 200, 255: 0}
#     for k, v in ind.items():
#         mask_disc[mask == k] = v
#     labeled, label_num = label(mask, connectivity=1, background=255, return_num=True)  # label_num = 2
#     image_props = regionprops(labeled, cache=False)
#     dis_trf = ndimage.distance_transform_edt(255 - mask)
#     adaptive_cup_dis_weight = np.zeros(mask.shape, dtype=np.float32)
#     adaptive_disc_dis_weight = np.zeros(mask.shape, dtype=np.float32)
#     adaptive_cup_dis_weight = adaptive_cup_dis_weight + (mask_cup / 255) * weight_cof
#     adaptive_disc_dis_weight = adaptive_disc_dis_weight + (mask_disc / 255) * weight_cof
#     adaptive_bck_dis_weight = np.ones(mask.shape, dtype=np.float32)
#
#     for num in range(1, label_num + 1):
#         image_prop = image_props[num - 1]
#         bool_dis = np.zeros(image_prop.image.shape)
#         bool_dis[image_prop.image] = 1.0
#         (min_row, min_col, max_row, max_col) = image_prop.bbox
#         temp_dis = dis_trf[min_row: max_row, min_col: max_col] * bool_dis
#
#         adaptive_cup_dis_weight[min_row: max_row, min_col: max_col] = adaptive_cup_dis_weight[min_row: max_row,
#                                                                       min_col: max_col] + get_bck_dis_weight(
#             temp_dis) * bool_dis
#         adaptive_disc_dis_weight[min_row: max_row, min_col: max_col] = adaptive_disc_dis_weight[min_row: max_row,
#                                                                        min_col: max_col] + get_bck_dis_weight(
#             temp_dis) * bool_dis
#         adaptive_bck_dis_weight[min_row: max_row, min_col: max_col] = adaptive_bck_dis_weight[min_row: max_row,
#                                                                       min_col: max_col] + get_bck_dis_weight(
#             temp_dis) * bool_dis
#
#     # get weight map for loss
#     bck_maxinum = np.max(adaptive_bck_dis_weight)
#     bck_mininum = np.min(adaptive_bck_dis_weight)
#     adaptive_bck_dis_weight = (adaptive_bck_dis_weight - bck_mininum) / (bck_maxinum - bck_mininum)
#
#     obj_maxinum = np.max(adaptive_disc_dis_weight)
#     obj_mininum = np.min(adaptive_disc_dis_weight)
#     adaptive_disc_dis_weight = (adaptive_disc_dis_weight - obj_mininum) / (obj_maxinum - obj_mininum)
#     adaptive_cup_dis_weight = (adaptive_cup_dis_weight - obj_mininum) / (obj_maxinum - obj_mininum)
#
#     adaptive_cup_dis_weight = adaptive_cup_dis_weight[np.newaxis, :, :]
#     adaptive_disc_dis_weight = adaptive_disc_dis_weight[np.newaxis, :, :]
#     adaptive_bck_dis_weight = adaptive_bck_dis_weight[np.newaxis, :, :]
#     adaptive_dis_weight = np.concatenate((adaptive_bck_dis_weight, adaptive_cup_dis_weight, adaptive_disc_dis_weight),axis=0)
#
#     return adaptive_dis_weight
#
# def get_bck_dis_weight(dis_map, w0=10, eps=1e-20):
#     max_dis = np.amax(dis_map)
#     std = max_dis / 2.58 + eps
#     weight_matrix = w0 * np.exp(-1 * pow((max_dis - dis_map), 2) / (2 * pow(std, 2)))
#     return weight_matrix

def calclulate_uncertainty(output, label):
    shape_uncertainty = (label * output) / sum(sum(label))
    shape_uncertainty += (1-label) * output / sum(sum(label))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
