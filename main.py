import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import argparse
from load_data import CVC_ClinicDB_trnDataSet, CVC_ClinicDB_tstDataSet
import network
import loss
import matplotlib
import matplotlib.pyplot as plt

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--train_data", type=str, default=None,
                        help="path to Dataset")
    parser.add_argument("--train_label", type=str, default=None,
                        help="path to Dataset")
    parser.add_argument("--test_data", type=str, default=None,
                        help="path to Dataset")
    parser.add_argument("--test_label", type=str, default=None,
                        help="path to Dataset")
    parser.add_argument("--train_list_path", type=str, default=None,
                        help="path to Dataset")
    parser.add_argument("--test_list_path", type=str, default=None,
                        help="path to Dataset")

    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: 2)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=True,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--lr1", type=float, default=0.01,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr2", type=float, default=0.01,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=300)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 1)')
    parser.add_argument("--crop_size", type=int, default=(256, 256))

    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=2,
                        help="random seed (default: 1)")
    return parser



def CreateTrnDataLoader(args, is_shuffle=True):
    img_mean = np.array((128, 128, 128), dtype=np.float32)
    source_dataset = CVC_ClinicDB_trnDataSet(args.train_data, args.train_label, args.train_list_path,
                                             crop_size=args.crop_size, mean=img_mean)
    source_dataloader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=is_shuffle, pin_memory=True,
                                        num_workers=4, drop_last=False)
    return source_dataset, source_dataloader


def CreateTstDataLoader(args):
    img_mean = np.array((128, 128, 128), dtype=np.float32)
    test_dataset = CVC_ClinicDB_tstDataSet(args.test_data, args.test_label, args.test_list_path,
                                           crop_size=args.crop_size, mean=img_mean)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True,
                                      num_workers=2, drop_last=False)
    return test_dataset, test_dataloader

def setup_model(opts):
    # =====  Set up model  ==========  Set up model  ==========  Set up model  ==========  Set up model  ==========  Set up model  =====
    # (all models are 'constructed at network.modeling)
    CKPT_PATH = "./network/best_deeplabv3plus_resnet101_voc_os16.pth"
    model1 = network.modeling.deeplabv3plus_resnet101(num_classes=21, output_stride=opts.output_stride)
    model1.load_state_dict(torch.load(CKPT_PATH)['model_state'])
    model = network.modeling.deeplabv3plus_resnet101(num_classes=opts.num_classes, output_stride=opts.output_stride)
    model1.classifier = model.classifier
    del model
    network.convert_to_separable_conv(model1.classifier)
    network.utils.set_bn_momentum(model1.backbone, momentum=0.01)

    CKPT_PATH = "./network/best_deeplabv3plus_mobilenet_voc_os16.pth"
    model2 = network.modeling.deeplabv3plus_mobilenet(num_classes=21, output_stride=opts.output_stride)
    model2.load_state_dict(torch.load(CKPT_PATH)['model_state'])
    model = network.modeling.deeplabv3plus_mobilenet(num_classes=opts.num_classes, output_stride=opts.output_stride)
    model2.classifier = model.classifier
    del model
    network.convert_to_separable_conv(model2.classifier)
    network.utils.set_bn_momentum(model2.backbone, momentum=0.01)

    optimizer1 = torch.optim.SGD(params=[
        {'params': model1.backbone.parameters(), 'lr': opts.lr1},
        {'params': model1.classifier.parameters(), 'lr': 0.1 * opts.lr1},
    ], lr=opts.lr1, momentum=0.9, weight_decay=opts.weight_decay)
    optimizer1.param_groups[0]['momentum'] = 0.95
    optimizer1.param_groups[1]['momentum'] = 0.99
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=opts.step_size, gamma=0.6)

    optimizer2 = torch.optim.SGD(params=[
        {'params': model2.backbone.parameters(), 'lr':opts.lr2},
        {'params': model2.classifier.parameters(), 'lr': 0.1 * opts.lr2},
    ], lr=opts.lr2, momentum=0.9, weight_decay=opts.weight_decay)
    optimizer2.param_groups[0]['momentum'] = 0.95
    optimizer2.param_groups[1]['momentum'] = 0.99
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opts.step_size, gamma=0.6)



    return model1, optimizer1, scheduler1, model2, optimizer2, scheduler2

def clean_select(Loss_clean1,Loss_clean2,alpha):
    clean_id1 = np.zeros(int(alpha*len(Loss_clean1)))
    clean_id2 = np.zeros(int(alpha*len(Loss_clean2)))
    noise_id = []
    Loss_clean1 = np.array(Loss_clean1)
    Loss_clean2 = np.array(Loss_clean2)
    index = np.argsort(-Loss_clean1)
    clean_id1 = index[len(clean_id1):]
    index = np.argsort(-Loss_clean2)
    clean_id2 = index[len(clean_id2):]
    for k in range(len(Loss_clean1)):
        if not k in clean_id1:
            if not k in clean_id2:
                noise_id.append(k)

    return clean_id1, clean_id2, noise_id

def clean_select_new(Loss_clean1, Loss_clean2):
    clean_id1 = []
    clean_id2 = []
    noise_id = []
    Loss_clean1 = np.array(Loss_clean1)
    Loss_clean2 = np.array(Loss_clean2)
    flag1 = (Loss_clean1 <= np.mean(Loss_clean1))
    flag2 = (Loss_clean2 <= np.mean(Loss_clean2))
    for i in range(len(Loss_clean1)):
        if flag1[i]:
            clean_id1.append(i)
        if flag2[i]:
            clean_id2.append(i)

    for k in range(len(Loss_clean1)):
        if not k in clean_id1:
            if not k in clean_id2:
                noise_id.append(k)

    return clean_id1, clean_id2, noise_id

def warmup(total_epoch, model, optimizer, scheduler, trainloader,testloader,device):
    #single net
    model.cuda()
    model.to(device)
    model = nn.DataParallel(model)
    cur_itrs = 0
    interval_loss = 0.0
    for epoch in range(total_epoch):
        print('Epoch {}/{}'.format(epoch, total_epoch - 1))
        print('-' * 10)
        # train
        model.train()
        print("lr1 = %f, lr2 = %f" % (optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
        for step_t, (t_x, t_y, name) in enumerate(trainloader):
            cur_itrs += 1
            optimizer.zero_grad()
            t_x = t_x.float().to(device)
            t_y = t_y.long().to(device)
            outputs = model(t_x)
            # softmax_outputs = F.softmax(outputs, dim=1)

            loss_clean = loss.myCrossEntropy2d(outputs, t_y) + 20 * loss.myDiceLoss(outputs, t_y, p=2) + 30 * loss.Smoothloss(outputs, t_y)
            loss_clean.backward()
            optimizer.step()
            scheduler.step()

            np_loss = loss_clean.detach().cpu().numpy()
            interval_loss += np_loss
            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" % (epoch, cur_itrs, total_epoch * len(trainloader), interval_loss))
                interval_loss = 0.0

        # validate
        print("validate...")


        total_dice = 0
        model.eval()
        for step_v, (v_x, v_y, name) in enumerate(testloader):

            v_x = v_x.float().to(device)
            outputs = model(v_x)
            outputs = nn.functional.softmax(outputs, dim=1)
            outputs = nn.functional.interpolate(outputs, (v_y.shape[1], v_y.shape[2]), mode='bilinear',
                                                align_corners=True).cpu().data[
                0].numpy()

            preds = np.asarray(np.argmax(outputs, axis=0), dtype=np.uint8)
            # preds = preds.transpose(1, 2, 0)
            val_dice= loss.calculate_dice(preds, v_y[0])
            #disc_mIOU, cup_mIOU = loss.calculate_mIOU(preds, v_y[0])
            total_dice += val_dice

        print("total_dice=%f" %
              (total_dice / len(testloader)))
    return model, optimizer, scheduler

def main(opts):
    ##setup gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"
    flag = torch.cuda.is_available()
    print("GPU availabel:", flag)
    device = torch.device('cuda')
    print(torch.cuda.device_count())
    print("Device: %s" % device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())

    _, trainloader = CreateTrnDataLoader(opts)
    print("Train dataset:", len(trainloader))
    _, testloader = CreateTstDataLoader(opts)
    print("Test dataset:", len(testloader))

    model1, optimizer1, scheduler1, model2, optimizer2, scheduler2 = setup_model(opts)

    if torch.cuda.device_count() > 1:
        # 如果不用os.environ的话，GPU的可见数量仍然是包括所有GPU的数量
        # 但是使用的还是只用指定的device_ids的GPU
        print("Let's use GPUs!")



    # #load pretrained model
    # warmup_net1 = './saved_model/warmup_res_high0.1.pth'
    # checkpoint = torch.load(warmup_net1, map_location=torch.device('cuda'))
    # model1.to(device)
    # model1 = nn.DataParallel(model1)
    # model1.cuda()
    # model1.load_state_dict(checkpoint["model_state"])
    # optimizer1.load_state_dict(checkpoint["optimizer_state"])
    # scheduler1.load_state_dict(checkpoint["scheduler_state"])
    # print("Model restored from %s" %warmup_net1)
    # print("train...,lr1 = %f, lr2 = %f" % (optimizer1.param_groups[0]['lr'], optimizer1.param_groups[1]['lr']))
    # del checkpoint  # free memory
    # warmup_net2 = './saved_model/warmup_mob_high0.1.pth'
    # checkpoint = torch.load(warmup_net2, map_location=torch.device('cuda'))
    # model2.to(device)
    # model2 = nn.DataParallel(model2)
    # model2.cuda()
    # model2.load_state_dict(checkpoint["model_state"])
    # optimizer2.load_state_dict(checkpoint["optimizer_state"])
    # scheduler2.load_state_dict(checkpoint["scheduler_state"])
    # print("Model restored from ./saved_model/warmup_mob_high0.9.pth")
    # print("train...,lr1 = %f, lr2 = %f" % (optimizer1.param_groups[0]['lr'], optimizer1.param_groups[1]['lr']))
    # del checkpoint  # free memory
    #
    # model1.to(device)
    # model1 = nn.DataParallel(model1)
    # model1.cuda()
    #
    # model2.to(device)
    # model2 = nn.DataParallel(model2)
    # model2.cuda()
    # optimizer1.param_groups[0]['lr'] = 0.005
    # optimizer1.param_groups[1]['lr'] = 0.0005
    # optimizer2.param_groups[0]['lr'] = 0.005
    # optimizer2.param_groups[1]['lr'] = 0.0005
    # =====  train loop  ==========  train loop  ==========  train loop  ==========  train loop  ==========  train loop  =====
    # co training
    epoch=0
    alpha=0.5
    while epoch<100:
        epoch = epoch +1
        for step_t, (t_x, t_y, name) in enumerate(trainloader):
            print("**************Epoch=", epoch, ",step=", step_t, "/", len(trainloader), '**********')
            # select clean samples
            model1.eval()
            model2.eval()
            t_x = t_x.float().to(device)
            t_y = t_y.long().to(device)
            # with torch.no_grad():
            #     # outputs1 = model1(t_x) #N,C,H,W
            #     # outputs2 = model2(t_x)
            #     n, c, h, w = t_x.size()
            #     Loss1 = []
            #     Loss2 = []
            #     for k in range(n):
            #         output = torch.zeros([1,c,h,w])
            #         output = output.float().to(device)
            #         label = torch.zeros([1, h, w])
            #         label = label.long().to(device)
            #         label[0] = t_y[k]
            #         output[0] = t_x[k]
            #         output = model1(output)
            #         loss_clean = loss.myCrossEntropy2d(output, label) + 20 * loss.myDiceLoss(output, label, p=2) + 30 * loss.Smoothloss(output, label)
            #         Loss1.append(loss_clean.cpu().numpy())
            #
            #         output = torch.zeros([1,c,h,w])
            #         output = output.float().to(device)
            #         output[0] = t_x[k]
            #         output = model2(output)
            #         loss_clean = loss.myCrossEntropy2d(output, label) + 20 * loss.myDiceLoss(output, label, p=2) + 30 * loss.Smoothloss(output, label)
            #         Loss2.append(loss_clean.cpu().numpy())
            #
            # #    print("Loss1:", *Loss1)
            # #    print("Loss2:", *Loss2)
            #     clean_id1, clean_id2, noise_id = clean_select_new(Loss1, Loss2)  ##考虑batchnorm=16

            # print("train...,lr1 = %f, lr2 = %f" % (optimizer1.param_groups[0]['lr'], optimizer1.param_groups[1]['lr']))
        #    print("clean_id1:",clean_id1)
        #    print("clean_id2:",clean_id2)
        #    print("noise_id:", noise_id)
            # co-training
            model1.train()
            model2.eval()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # update network1

            clean_samples = torch.zeros([len(clean_id2), c, h, w])
            clean_label = torch.zeros([len(clean_id2), h, w])
            clean_samples = clean_samples.float().to(device)
            clean_label = clean_label.long().to(device)
            for i in range(len(clean_id2)):
                clean_samples[i] = t_x[clean_id2[i]]
                clean_label[i] = t_y[clean_id2[i]]
            # for i in range(len(clean_id2)):
            #     clean_samples[n-i-1] = t_x[clean_id2[i]]
            #     clean_label[n-i-1] = t_y[clean_id2[i]]

            outputs1 = model1(clean_samples)  # N,C,H,W
            loss_clean1 = loss.myCrossEntropy2d(outputs1, clean_label) + 20 * loss.myDiceLoss(outputs1, clean_label, p=2, smooth=1e-8) + 30 * loss.Smoothloss(outputs1, clean_label)
            print("loss_clean1: ",loss_clean1)

            loss_clean1.backward()
            optimizer1.step()
            scheduler1.step()
            # update network2
            model2.train()
            model1.eval()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            clean_samples = torch.zeros([len(clean_id1), c, h, w])
            clean_label = torch.zeros([len(clean_id1), h, w])
            clean_samples = clean_samples.float().to(device)
            clean_label = clean_label.long().to(device)
            for i in range(len(clean_id1)):
                clean_samples[i] = t_x[clean_id1[i]]
                clean_label[i] = t_y[clean_id1[i]]
            # for i in range(len(clean_id1)):
            #     clean_samples2[n-i-1] = t_x[clean_id1[i]]
            #     clean_label2[n-i-1] = t_y[clean_id1[i]]

            outputs2 = model2(clean_samples)  # N,C,H,W
            loss_clean2 = loss.myCrossEntropy2d(outputs2, clean_label) + 20 * loss.myDiceLoss(outputs2, clean_label, p=2, smooth=1e-8) + 30 * loss.Smoothloss(outputs2, clean_label)
            print("loss_clean2: ", loss_clean2)
            loss_clean2.backward()
            optimizer2.step()
            scheduler2.step()

            #label cleansing
            if epoch>9 and len(noise_id)>=2:
                print("label cleansing running")

                noisy_samples = torch.zeros([len(noise_id), c, h, w])
                noisy_label = torch.zeros([len(noise_id), h, w])
                noisy_samples = noisy_samples.float().to(device)
                noisy_label = noisy_label.long().to(device)
                for i in range(len(noise_id)):
                    noisy_samples[i] = t_x[noise_id[i]]
                    noisy_label[i] = t_y[noise_id[i]]
                # optimizer1.zero_grad()
                # optimizer2.zero_grad()
                # outputs1 = model1(noisy_samples)  # N,C,H,W
                # outputs2 = model2(noisy_samples)
                # preds1 = np.asarray(np.argmax(outputs1.detach().cpu(), axis=1), dtype=np.int64)
                # preds1 = torch.from_numpy(preds1)
                # preds1 = preds1.cuda()
                # preds1.requires_grad = False
                # preds1.cuda()
                # loss_noise2 = loss.myCrossEntropy2d(outputs2, preds1) + 20 * loss.myDiceLoss(outputs2, preds1, p=2, smooth=1e-8) + 30 * loss.Smoothloss(outputs2, preds1)
                # loss_noise2.backward()
                # optimizer2.step()
                # scheduler2.step()
                # print("loss_noise2: ", loss_noise2)

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                outputs1 = model1(noisy_samples)  # N,C,H,W
                outputs2 = model2(noisy_samples)
                preds2 = np.asarray(np.argmax(outputs2.detach().cpu(), axis=1), dtype=np.int64)
                preds2 = torch.from_numpy(preds2)
                preds2 = preds2.cuda()
                preds2.requires_grad = False
                loss_noise1 = loss.myCrossEntropy2d(outputs1, preds2) + 20 * loss.myDiceLoss(outputs1, preds2, p=2, smooth=1e-8) + 30 * loss.Smoothloss(outputs1, preds2)
                print("loss_noise1: ", loss_noise1)
                loss_noise1.backward()
                optimizer1.step()
                scheduler1.step()





        #     # validate
        print("validate...")
        torch.save({
            "model_state": model1.module.state_dict(),
            "optimizer_state": optimizer1.state_dict(),
            "scheduler_state": scheduler1.state_dict(),
        }, './saved_model/co_high_0.1.pth')
        # torch.save({
        #     "cur_itrs": cur_itrs,
        #     "model_state": model2.module.state_dict(),
        #     "optimizer_state": optimizer2.state_dict(),
        #     "scheduler_state": scheduler2.state_dict(),
        # }, './saved_model/High_noisy/warmup_0.5/co_training.pth')

        total_dice = 0
        model1.eval()
        # model2.eval()
        for step_v, (v_x, v_y, name) in enumerate(testloader):

            v_x = v_x.float().to(device)
            # v_y = v_y.long().to(device)
            outputs = model1(v_x)
            outputs = nn.functional.softmax(outputs, dim=1)
            outputs = nn.functional.interpolate(outputs, (v_y.shape[1], v_y.shape[2]), mode='bilinear',
                                                align_corners=True).cpu().data[
                0].numpy()

            preds = np.asarray(np.argmax(outputs, axis=0), dtype=np.uint8)
            # preds = preds.transpose(1, 2, 0)
            val_dice = loss.calculate_dice(preds, v_y[0])
            # disc_mIOU, cup_mIOU = loss.calculate_mIOU(preds, v_y[0])
            total_dice += val_dice


        print("total_dice=%f" %
              (total_dice / len(testloader)))



if __name__ == '__main__':
    opts = get_argparser().parse_args()
    opts.train_data = './dataset/Original'
    opts.train_label = './dataset/using_masks'
    opts.train_list_path = './dataset/train_list.txt'
    opts.test_data = './dataset/Original'
    opts.test_label = './dataset/Ground Truth'
    opts.test_list_path = './dataset/test_list.txt'
    opts.num_classes = 2
    opts.batch_size = 1
    # filelist = os.listdir(opts.train_data)
    # test_list = random.sample(filelist,122)
    # train_list = list(set(filelist).difference(test_list))
    # file = open('F:\Clinic_noise/train_list.txt','a')
    # for str in train_list:
    #     file.write(str+'\n')
    # file.close()
    # file = open('F:\Clinic_noise/test_list.txt', 'a')
    # for str in test_list:
    #     file.write(str + '\n')
    # file.close()

    main(opts)
