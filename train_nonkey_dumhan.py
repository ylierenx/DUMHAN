import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model_dumhan
import dataloder
from math import log10
import sys
from os.path import join
from os import listdir
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor
import cv2
import matplotlib.image as mpimg

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

flag = True
device = torch.device("cuda:0")

# key reconstruction
sample_and_inirecon_key = model_dumhan.sample_and_inirecon_key(num_filters=256, B_size=32, device=device)
sample_and_inirecon_key.to(device)
# nonkey reconstruction
sample_and_recon_nonkey = model_dumhan.sample_and_inirecon_nonkey(num_filters=102, B_size=32, device=device)
sample_and_recon_nonkey.to(device)

trainset = dataloder2.UCF101(gop_size=9, image_size=160)
train_loader = dataloder2.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

params_nonkey = list(sample_and_recon_nonkey.parameters())
optimizer_nonkey = optim.Adam(params_nonkey, lr=0.0001)

if flag:
    dict1 = torch.load('./check_point/key_0.5_dumhan.ckpt')
    sample_and_inirecon_key.load_state_dict(dict1['state_dict_sample_and_inirecon'])
else:
    dict1 = {'epoch': -1}

checkpoint_counter = 1


def load_img(filepath):
    img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
    img = np.pad(img, ((0, 16), (0, 16)), 'constant', constant_values=0)
    return img


def get_single_image(image_path):
    image = load_img(image_path)
    input_compose = Compose([ToTensor()])
    image = input_compose(image)
    return image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "tif"])


def PSNR(data, tar):
    data = torch.squeeze(data)
    tar = torch.squeeze(tar)
    data = data.cpu().numpy()
    tar = tar.cpu().numpy()
    data = data[0:144, 0:176]
    tar = tar[0:144, 0:176]
    data = data.astype(np.float32)
    tar = tar.astype(np.float32)
    max_gray = 1.
    mse = np.mean(np.power(data - tar, 2))
    return 10. * np.log10(max_gray ** 2 / mse)


def test():
    key_temp_t = torch.zeros(1, 512, 1, 1)
    nonkey_temp_t = torch.zeros(1, 512, 1, 1)
    key_temp_t[:, 0:512, :, :] = 1
    nonkey_temp_t[:, 0:102, :, :] = 1
    key_temp_t = key_temp_t.to(device)
    nonkey_temp_t = nonkey_temp_t.to(device)
    for video_name in sorted(os.listdir('./test_video')):
        psnr_ave = []
        file_path = './test_video/' + video_name
        gop_size = 8
        image_width = 176
        image_height = 144
        input_tensor = torch.zeros([1, gop_size + 1, 1, image_height + 16, image_width + 16])
        file = listdir(file_path)
        file.sort()
        image_filenames = [join(file_path, x) for x in file if is_image_file(x)]
        num_gop = len(image_filenames) / gop_size
        img_index = 0
        with torch.no_grad():
            for i in range(12):
                if img_index == (len(image_filenames) - 8):
                    break
                # start = time.time()
                for j in range(int(gop_size) + 1):
                    s = "%06d" % (j + i * gop_size)
                    image_name = s + '.tif'
                    image_path = os.path.join(file_path, image_name)
                    single_image = get_single_image(image_path)
                    input_tensor[0, j, 0, :, :] = single_image

                frame0 = input_tensor[:, 0, :, :, :]
                frame8 = input_tensor[:, 8, :, :, :]
                F_0 = frame0.to(device)
                F_8 = frame8.to(device)
                # key_initial reconstruction
                F_08 = torch.cat((F_0, F_8), dim=0)
                out_F_08 = sample_and_inirecon_key(x_ini=F_08)
                out_F_0 = out_F_08[[0], :, :, :]
                out_F_8 = out_F_08[[1], :, :, :]
                out_F_0 = out_F_0.detach()
                out_F_8 = out_F_8.detach()
                psnr_temp = PSNR(out_F_0, F_0)
                psnr_ave.append(psnr_temp)
                img_index = img_index + 1
                for k in range(int(gop_size) - 1):
                    frame1 = input_tensor[:, k + 1, :, :, :]
                    F_1 = frame1.to(device)
                    # non_key_initial reconstruction
                    out_F_1 = sample_and_recon_nonkey(F_1, out_F_0, out_F_8)
                    psnr_temp = PSNR(out_F_1, F_1)
                    psnr_ave.append(psnr_temp)
                # sys.exit()
            psnr_ave = psnr_ave[0:96]
            psnr_ave_m = np.mean(psnr_ave)
    return psnr_ave_m, psnr_ave


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    for epoch in range(0, 221):
        print("Epoch: ", epoch)

        if (epoch >= 200) and (epoch < 210):
            lr = 0.00001
            adjust_learning_rate(optimizer_nonkey, lr)
        if epoch >= 210:
            lr = 0.000001
            adjust_learning_rate(optimizer_nonkey, lr)

        start = time.time()
        for i, (inputs, f_index) in enumerate(train_loader):

            frame0 = inputs[:, 0, :, :, :]
            frame1 = inputs[:, 1, :, :, :]
            frame8 = inputs[:, 2, :, :, :]
            F_0 = frame0.to(device)
            F_1 = frame1.to(device)
            F_8 = frame8.to(device)
            optimizer_nonkey.zero_grad()

            # key_initial reconstruction
            with torch.no_grad():

                out_F_0 = sample_and_inirecon_key(x_ini=F_0)
                out_F_8 = sample_and_inirecon_key(x_ini=F_8)
                # F_08 = torch.cat((F_0, F_8), dim=0)
                # out_F_08 = sample_and_inirecon_key(x_ini=F_08)
                # out_F_0 = out_F_08[:batch_size, :, :, :]
                # out_F_8 = out_F_08[batch_size:, :, :, :]

            out_F_1 = sample_and_recon_nonkey(F_1, out_F_0, out_F_8)

            recnLoss_F1 = torch.mean(
                torch.norm((F_1 - out_F_1), p=2, dim=(2, 3)) * torch.norm((F_1 - out_F_1), p=2, dim=(2, 3)))
            recnLoss_F0 = torch.mean(
                torch.norm((F_0 - out_F_0), p=2, dim=(2, 3)) * torch.norm((F_0 - out_F_0), p=2, dim=(2, 3)))
            recnLoss_F8 = torch.mean(
                torch.norm((F_8 - out_F_8), p=2, dim=(2, 3)) * torch.norm((F_8 - out_F_8), p=2, dim=(2, 3)))

            all_loss = recnLoss_F1
            all_loss.backward()
            optimizer_nonkey.step()

            if ((i % 100) == 0):
                print(f_index + 1)
                print(
                    " L0: %0.6f  L_i: %0.6f  L8: %0.6f %d" % (
                        recnLoss_F0.item(), recnLoss_F1.item(), recnLoss_F8.item(), epoch))
                f = open('train_0.5_0.1_dumhan.txt', 'a')
                f.write(" %0.6f %d " % (recnLoss_F1.item(), epoch))
                f.write('\n')
                f.close()
                psnr_ave_m, psnr_ave = test()
                print('test')
                print(psnr_ave[0:16])
                print(psnr_ave_m)
                # sys.exit()
                end = time.time()
                print(end - start)
                start = time.time()

        end = time.time()
        print(end - start)
        # if ((epoch % 10) == 0 and epoch > 500):
        if ((epoch % 10) == 0 and epoch > 0 and epoch < 200):
            dict1 = {
                'Detail': "dumhan",
                'epoch': epoch,
                'state_dict_sample_and_inirecon_key': sample_and_inirecon_key.state_dict(),
                'state_dict_optimizer_nonkey': optimizer_nonkey.state_dict(),
                'state_dict_sample_and_recon_nonkey': sample_and_recon_nonkey.state_dict()
            }
            torch.save(dict1, './check_point' + "/dumhan_0.5_0.1_" + str(checkpoint_counter) + ".ckpt")
            checkpoint_counter += 1
        else:
            if ((epoch % 10) == 0 and epoch >= 200 and epoch < 221):
                dict1 = {
                    'Detail': "dumhan",
                    'epoch': epoch,
                    'state_dict_sample_and_inirecon_key': sample_and_inirecon_key.state_dict(),
                    'state_dict_optimizer_nonkey': optimizer_nonkey.state_dict(),
                    'state_dict_sample_and_recon_nonkey': sample_and_recon_nonkey.state_dict()
                }
                torch.save(dict1, './check_point' + "/dumhan_0.5_0.1_" + str(checkpoint_counter) + ".ckpt")
                checkpoint_counter += 1
