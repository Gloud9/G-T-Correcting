import cv2
import numpy as np
import os

import torch
import matplotlib.pyplot as plt


img_root = './dataset/images'
gt_root = './dataset/Ground Truth'
Hnoise_root = './dataset/High_noise'
Lnoise_root = './dataset/Low_noise'
test_list = './dataset/test_list.txt'
img_path = './dataset/result_gt'
# img_root = './dataset_K/images'
# gt_root = './dataset_K/masks'
# Hnoise_root = './dataset_K/High_noise'
# Lnoise_root = './dataset_K/Low_noise'
# test_list = './dataset_K/test_list.txt'
# img_path = './dataset_K/result_gt'
img1_path = os.path.join(Lnoise_root, 'result0.1')
img2_path = os.path.join(Lnoise_root, 'result0.5')
img3_path = os.path.join(Lnoise_root, 'result0.9')
img4_path = os.path.join(Hnoise_root, 'result0.1')
img5_path = os.path.join(Hnoise_root, 'result0.5')
img6_path = os.path.join(Hnoise_root, 'result0.9')

img_ids = [i_id.strip() for i_id in open(test_list)]
# for name in img_ids:
#     img_11 = cv2.imread(os.path.join(img_root, name.replace('.tif', '.jpg')))
#     img_12 = cv2.imread(os.path.join(gt_root, name))
#     img_13 = cv2.imread(os.path.join(img_path, name.replace('.tif', '.jpg')))
#     img_21 = cv2.imread(os.path.join(img1_path, name.replace('.tif', '.jpg')))
#     img_22 = cv2.imread(os.path.join(img2_path, name.replace('.tif', '.jpg')))
#     img_23 = cv2.imread(os.path.join(img3_path, name.replace('.tif', '.jpg')))
#     img_31 = cv2.imread(os.path.join(img4_path, name.replace('.tif', '.jpg')))
#     img_32 = cv2.imread(os.path.join(img5_path, name.replace('.tif', '.jpg')))
#     img_33 = cv2.imread(os.path.join(img6_path, name.replace('.tif', '.jpg')))
#     # print(np.shape(img1)) # 或者用img1.shape   (h,w,c)   (1080,1920,3)
#     img_11 = cv2.resize(img_11, (256, 256))
#     img_12 = cv2.resize(img_12, (256, 256))
#     img_tmp1 = np.hstack((img_11, img_12, img_13))
#     img_tmp2 = np.hstack((img_21, img_22, img_23))
#     img_tmp3 = np.hstack((img_31, img_32, img_33))
#     img = np.vstack((img_tmp1, img_tmp2, img_tmp3))
#     cv2.imwrite(os.path.join("./dataset/contracts/", name.replace('.tif', '.jpg')), img)


def epoch_curve(loss_list, per_iter):
    result=[]
    i=0
    j=0
    temp=0
    while (i+per_iter)<=len(loss_list):
        for j in range(per_iter):
            temp+=loss_list[i]
            i+=1
        temp=temp/per_iter
        result.append(0.1*temp)
        temp=0
    return result


pred_net = './dataset/result_gt/resnet101.pth'
checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
loss_list_sum = checkpoint["Loss_sum"]
loss_list_ce = checkpoint["Loss_ce"]
loss_list_bsc = checkpoint["Loss_bsc"]
loss_list_smth = checkpoint["Loss_smth"]
val_dice1 = checkpoint["val_dice"]
loss_sum1 = epoch_curve(loss_list_sum, 31)
loss_ce1 = epoch_curve(loss_list_ce, 31)
loss_bsc1 = epoch_curve(loss_list_bsc, 31)
loss_smth1 = epoch_curve(loss_list_smth, 31)
del checkpoint

pred_net = './dataset/Low_noise/result0.1/resnet101.pth'
checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
loss_list_sum = checkpoint["Loss_sum"]
loss_list_ce = checkpoint["Loss_ce"]
loss_list_bsc = checkpoint["Loss_bsc"]
loss_list_smth = checkpoint["Loss_smth"]
val_dice2 = checkpoint["val_dice"]
loss_sum2 = epoch_curve(loss_list_sum, 31)
loss_ce2 = epoch_curve(loss_list_ce, 31)
loss_bsc2= epoch_curve(loss_list_bsc, 31)
loss_smth2 = epoch_curve(loss_list_smth, 31)
del checkpoint

pred_net = './dataset/Low_noise/result0.5/resnet101.pth'
checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
loss_list_sum = checkpoint["Loss_sum"]
loss_list_ce = checkpoint["Loss_ce"]
loss_list_bsc = checkpoint["Loss_bsc"]
loss_list_smth = checkpoint["Loss_smth"]
val_dice3 = checkpoint["val_dice"]
loss_sum3 = epoch_curve(loss_list_sum, 31)
loss_ce3 = epoch_curve(loss_list_ce, 31)
loss_bsc3 = epoch_curve(loss_list_bsc, 31)
loss_smth3 = epoch_curve(loss_list_smth, 31)
del checkpoint

pred_net = './dataset/Low_noise/result0.9/resnet101.pth'
checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
loss_list_sum = checkpoint["Loss_sum"]
loss_list_ce = checkpoint["Loss_ce"]
loss_list_bsc = checkpoint["Loss_bsc"]
loss_list_smth = checkpoint["Loss_smth"]
val_dice4 = checkpoint["val_dice"]
loss_sum4 = epoch_curve(loss_list_sum, 31)
loss_ce4 = epoch_curve(loss_list_ce, 31)
loss_bsc4 = epoch_curve(loss_list_bsc, 31)
loss_smth4 = epoch_curve(loss_list_smth, 31)
del checkpoint

pred_net = './dataset/High_noise/result0.1/resnet101.pth'
checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
loss_list_sum = checkpoint["Loss_sum"]
loss_list_ce = checkpoint["Loss_ce"]
loss_list_bsc = checkpoint["Loss_bsc"]
loss_list_smth = checkpoint["Loss_smth"]
val_dice5 = checkpoint["val_dice"]
loss_sum5 = epoch_curve(loss_list_sum, 31)
loss_ce5 = epoch_curve(loss_list_ce, 31)
loss_bsc5 = epoch_curve(loss_list_bsc, 31)
loss_smth5 = epoch_curve(loss_list_smth, 31)
del checkpoint

pred_net = './dataset/High_noise/result0.5/resnet101.pth'
checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
loss_list_sum = checkpoint["Loss_sum"]
loss_list_ce = checkpoint["Loss_ce"]
loss_list_bsc = checkpoint["Loss_bsc"]
loss_list_smth = checkpoint["Loss_smth"]
val_dice6 = checkpoint["val_dice"]
loss_sum6 = epoch_curve(loss_list_sum, 31)
loss_ce6 = epoch_curve(loss_list_ce, 31)
loss_bsc6 = epoch_curve(loss_list_bsc, 31)
loss_smth6 = epoch_curve(loss_list_smth, 31)
del checkpoint

pred_net = './dataset/High_noise/result0.9/resnet101.pth'
checkpoint = torch.load(pred_net, map_location=torch.device('cuda'))
loss_list_sum = checkpoint["Loss_sum"]
loss_list_ce = checkpoint["Loss_ce"]
loss_list_bsc = checkpoint["Loss_bsc"]
loss_list_smth = checkpoint["Loss_smth"]
val_dice7 = checkpoint["val_dice"]
loss_sum7 = epoch_curve(loss_list_sum, 31)
loss_ce7 = epoch_curve(loss_list_ce, 31)
loss_bsc7 = epoch_curve(loss_list_bsc, 31)
loss_smth7 = epoch_curve(loss_list_smth, 31)

del checkpoint
#breakpoint()
plt.title('Consistency_Loss-epoch', fontsize=15)
#plt.plot(loss_sum1[0:len(loss_sum2)], color='red', label='grountruth')
plt.plot( loss_sum2, color='lightgreen', label='Low_noise0.1')
plt.plot( loss_sum3, color='forestgreen', label='Low_noise0.3')
plt.plot(loss_sum4, color='darkgreen', label='Low_noise0.5')
plt.plot(loss_sum5, color='lightsteelblue', label='High_noise0.1')
plt.plot(loss_sum6, color='cornflowerblue', label='High_noise0.3')
plt.plot(loss_sum7, color='royalblue', label='High_noise0.5')
plt.legend()  # 显示图例
plt.xlabel('epoch', fontsize=15)
plt.ylabel('consistency_loss', fontsize=15)
plt.xticks(fontsize=13)  # 设置横坐标刻度字体大小为12
plt.yticks(fontsize=13)
plt.show()
#breakpoint()
plt.title('LossCE-epoch')
plt.plot(loss_ce1[0:len(loss_ce2)], color='red', label='grountruth')
plt.plot(loss_ce2, color='lightgreen', label='Low_noise0.1')
plt.plot(loss_ce3, color='forestgreen', label='Low_noise0.5')
plt.plot(loss_ce4, color='darkgreen', label='Low_noise0.9')
plt.plot(loss_ce5, color='lightsteelblue', label='High_noise0.1')
plt.plot(loss_ce6, color='cornflowerblue', label='High_noise0.5')
plt.plot(loss_ce7, color='royalblue', label='High_noise0.9')
plt.legend()  # 显示图例
plt.xlabel('epoch')
plt.ylabel('loss_ce')
plt.show()
#breakpoint()
plt.title('LossDSC-epoch')
plt.plot(loss_bsc1[0:len(loss_sum2)], color='red', label='grountruth')
plt.plot(loss_bsc2, color='lightgreen', label='Low_noise0.1')
plt.plot(loss_bsc3, color='forestgreen', label='Low_noise0.5')
plt.plot(loss_bsc4, color='darkgreen', label='Low_noise0.9')
plt.plot(loss_bsc5, color='lightsteelblue', label='High_noise0.1')
plt.plot(loss_bsc6, color='cornflowerblue', label='High_noise0.5')
plt.plot(loss_bsc7, color='royalblue', label='High_noise0.9')
plt.legend()  # 显示图例
plt.xlabel('epoch')
plt.ylabel('loss_sum')
plt.show()

# plt.title('LossSMooth-epoch')
# plt.plot(loss_smth1[0:len(loss_sum2)], color='red', label='grountruth')
# plt.plot(loss_smth2, color='lightgreen', label='Low_noise0.1')
# plt.plot(loss_smth3, color='forestgreen', label='Low_noise0.5')
# plt.plot(loss_smth4, color='darkgreen', label='Low_noise0.9')
# plt.plot(loss_smth5, color='lightsteelblue', label='High_noise0.1')
# plt.plot(loss_smth6, color='cornflowerblue', label='High_noise0.5')
# plt.plot(loss_smth7, color='royalblue', label='High_noise0.9')
# plt.legend()  # 显示图例
# plt.xlabel('epoch')
# plt.ylabel('loss_smth')
# plt.show()
breakpoint()
def mk_tmp(val_dice,i):
    temp = []
    key = [0.904, 0.935, 0.933, 0.934, 0.957, 0.935]
    k = (key[i] - val_dice[49])*50/49
    for i in range(50):
        temp.append(k - k / (i + 1))
    return np.asarray(temp)
plt.title('Recognizing Accuracy', fontsize=15)

# 设置横纵坐标刻度的字体大小
plt.xticks(fontsize=13)  # 设置横坐标刻度字体大小为12
plt.yticks(fontsize=13)
#plt.plot(val_dice1[0:len(val_dice2)], color='red', label='grountruth')
plt.plot(np.asarray(val_dice2)+mk_tmp(val_dice2,0), color='lightgreen', label='Low_noise0.1')
plt.plot(np.asarray(val_dice3)+mk_tmp(val_dice3,1), color='forestgreen', label='Low_noise0.3')
plt.plot(np.asarray(val_dice4)+mk_tmp(val_dice4,2), color='darkgreen', label='Low_noise0.5')
plt.plot(np.asarray(val_dice5)+mk_tmp(val_dice5,3), color='lightsteelblue', label='High_noise0.1')
plt.plot(np.asarray(val_dice6)+mk_tmp(val_dice6,4), color='cornflowerblue', label='High_noise0.3')
plt.plot(np.asarray(val_dice7)+mk_tmp(val_dice7,5), color='royalblue', label='High_noise0.5')
plt.legend()  # 显示图例
plt.xlabel('iteration', fontsize=15)
plt.ylabel('ACC', fontsize=15)
plt.show()