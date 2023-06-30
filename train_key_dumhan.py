import math
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model_dumhan
import dataloder1
from math import log10
import sys
import time
from PIL import Image
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg
import scipy.io as scio
import random
from torchvision.transforms import Compose, ToTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device("cuda:0")
load_flag = False
sample_and_inirecon = model_dumhan.sample_and_inirecon_key(num_filters=512, B_size=32, device=device)
sample_and_inirecon.to(device)

params_key = list(sample_and_inirecon.parameters())

optimizer_key = optim.Adam(params_key, lr=0.0001)

trainset = dataloder1.UCF101(gop_size=1, image_size=160)
train_loader = dataloder1.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True,
                                     drop_last=True)

if load_flag:
    dict1 = torch.load('./check_point/key_0.5.ckpt')
    sample_and_inirecon.load_state_dict(dict1['state_dict_sample_and_inirecon'])
    optimizer_key.load_state_dict(dict1['state_dict_optimizer_key'])
else:
    dict1 = {'epoch': -1}


checkpoint_counter = 1


def rgb2ycbcr_y(rgb_img):
    img_y = np.round(rgb_img[:, :, 0] * 0.256789 + rgb_img[:, :, 1] * 0.504129 + rgb_img[:, :, 2] * 0.097906 + 16)
    img_y = (img_y / 255.)
    return img_y


def psnr(img_rec, img_ori):
    img_rec = img_rec.astype(np.float32)
    img_ori = img_ori.astype(np.float32)
    max_gray = 1.
    mse = np.mean(np.power(img_rec - img_ori, 2))
    return 10. * np.log10(max_gray ** 2 / mse)


def adjust_learning_rate1(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test_splnet(block_size=32):
    for dataset_name in sorted(os.listdir('./dataset1')):
        psnr_value = []
        fnames = []
        dataset_n = os.path.join('./dataset1', dataset_name)
        for file_name in sorted(os.listdir(dataset_n)):
            fnames.append(os.path.join(dataset_n, file_name))
    with torch.no_grad():
        for f_i in range(len(fnames)):
            img = Image.open(fnames[f_i])
            I = np.array(img)
            # if len(I.shape) == 3:
            #     I = rgb2ycbcr_y(I)

            I = Image.fromarray(I)

            input_compose = Compose([ToTensor()])
            I = input_compose(I)
            I = I.unsqueeze(0)

            inputs = I

            ih = I.shape[2]
            iw = I.shape[3]

            if np.mod(iw, block_size) != 0:
                col_pad = block_size - np.mod(iw, block_size)
                inputs = torch.cat((inputs, torch.zeros([1, 1, ih, col_pad])), axis=3)
            else:
                col_pad = 0
                inputs = inputs
            if np.mod(ih, block_size) != 0:
                row_pad = block_size - np.mod(ih, block_size)
                inputs = torch.cat((inputs, torch.zeros([1, 1, row_pad, iw + col_pad])), axis=2)
            else:
                row_pad = 0
            inputs = inputs.cuda()

            x_output = sample_and_inirecon(x_ini=inputs)

            x_output = x_output.cpu().numpy()
            I = I.cpu().numpy()
            recon_img = x_output[0, 0, :ih, :iw]
            ori_img = I[0, 0, :ih, :iw]
            p1 = psnr(recon_img, ori_img)
            psnr_value.append(p1)
    #print(psnr_value)
    #sys.exit()

    return np.mean(psnr_value)


if __name__ == '__main__':
    p1 = test_splnet()
    start = time.time()
    gb_flag = 0
    for epoch in range(0, 222):

        if epoch >= 210 and epoch < 215:
            lr = 0.00001
            adjust_learning_rate1(optimizer_key, lr)

        if epoch >= 215:
            lr = 0.000001
            adjust_learning_rate1(optimizer_key, lr)

        for i, inputs in enumerate(train_loader):


            inputs = inputs[:, 0, :, :, :]
            inputs = inputs.to(device)

            optimizer_key.zero_grad()

            x_output = sample_and_inirecon(x_ini=inputs)

            recnLoss_final = torch.mean(
                torch.norm((inputs - x_output), p=2, dim=(2, 3)) * torch.norm((inputs - x_output), p=2, dim=(2, 3)))


            recnLoss_all = recnLoss_final
            recnLoss_all.backward()
            optimizer_key.step()

            if ((i % 100) == 0):
                print('test')

                p1 = test_splnet()

                print("0.5:%0.6f" % (p1))

                print("train_loss: %0.6f train_loss_n: %0.6f Iterations: %4d/%4d epoch:%d " % (
                    recnLoss_final.item(), recnLoss_all.item(), i, len(train_loader), epoch))

                f = open('test_key_0.5_dumhan.txt', 'a')
                f.write("%0.6f %d %0.6f" % (p1, epoch, recnLoss_final.item()))
                f.write('\n')
                f.close()

                end = time.time()
                print(end - start)
                start = time.time()
        if ((epoch % 50) == 0 and epoch > 0 and epoch < 210):
            dict1 = {
                'Detail': "dumhan",
                'epoch': epoch,

                'state_dict_sample_and_inirecon': sample_and_inirecon.state_dict(),

                'state_dict_optimizer_key': optimizer_key.state_dict()

            }
            torch.save(dict1, './check_point' + "/key_0.5_dumhan_" + str(checkpoint_counter) + ".ckpt")
            checkpoint_counter += 1
        else:
            if ((epoch % 5) == 0 and epoch >= 210 and epoch < 222):
                dict1 = {
                    'Detail': "dumhan",
                    'epoch': epoch,

                    'state_dict_sample_and_inirecon': sample_and_inirecon.state_dict(),

                    'state_dict_optimizer_key': optimizer_key.state_dict()

                }
                torch.save(dict1, './check_point' + "/key_0.5_dumhan_" + str(checkpoint_counter) + ".ckpt")
                checkpoint_counter += 1













