import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import sys
import torchvision

num_of = 32
num_of1 = 16
num_of2 = 16


class backWarp_MH(nn.Module):

    def __init__(self, device):
        super(backWarp_MH, self).__init__()
        self.device = device

    def forward(self, img, flow, num_of=32):
        B, C, H, W = img.shape
        img = img.repeat(1, num_of, 1, 1)
        img = torch.reshape(img, (B * num_of, C, H, W))
        # img = img.view(B*num_of, C, H, W)
        Bf, Cf, Hf, Wf = flow.shape
        # flow = flow.view(Bf*num_of, Cf//num_of, Hf, Wf)
        flow = torch.reshape(flow, (Bf * num_of, Cf // num_of, Hf, Wf))
        # create a grid
        # print(H)
        self.gridX, self.gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        # print(np.shape(gridX))
        # sys.exit()
        self.gridX = torch.tensor(self.gridX, requires_grad=False, device=self.device)
        # self.gridX.to(device)
        self.gridY = torch.tensor(self.gridY, requires_grad=False, device=self.device)

        # self.gridY.to(device)
        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        imgOut = imgOut.view(B, C * num_of, H, W)

        return imgOut


class synthes_net(nn.Module):
    def __init__(self):
        super(synthes_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of + 1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=64 + 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        xa = F.relu(x, inplace=True)
        x = F.interpolate(xa, scale_factor=0.5, mode='bilinear')

        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        xb = F.relu(x, inplace=True)
        x = F.interpolate(xb, scale_factor=0.5, mode='bilinear')

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)

        up1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xb, up1), dim=1)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)

        up2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xa, up2), dim=1)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        output = self.conv10(x)

        # output = x1 + output

        return output


class FLOWNET2_key(nn.Module):
    def __init__(self):
        super(FLOWNET2_key, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of * 3 + 1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=64 + 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=num_of * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        # x = torch.cat((x1, x2, X3, x4), dim=1)
        x1 = x1.repeat(1, num_of + 1, 1, 1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        xa = F.relu(x, inplace=True)
        x = F.interpolate(xa, scale_factor=0.5, mode='bilinear')

        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        xb = F.relu(x, inplace=True)
        x = F.interpolate(xb, scale_factor=0.5, mode='bilinear')

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)

        up1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xb, up1), dim=1)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)

        up2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xa, up2), dim=1)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        output = self.conv10(x)

        return output


class FLOWNET1_key(nn.Module):
    def __init__(self):
        super(FLOWNET1_key, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of + 1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=64 + 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=num_of * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x = torch.cat((x1, x2, X3), dim=1)
        x = x.repeat(1, num_of + 1, 1, 1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        xa = F.relu(x, inplace=True)
        x = F.interpolate(xa, scale_factor=0.5, mode='bilinear')

        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        xb = F.relu(x, inplace=True)
        x = F.interpolate(xb, scale_factor=0.5, mode='bilinear')

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)

        up1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xb, up1), dim=1)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)

        up2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xa, up2), dim=1)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        output = self.conv10(x)

        return output


class basic_block_b_key(nn.Module):
    def __init__(self):
        super(basic_block_b_key, self).__init__()
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.flownet = FLOWNET2_key()
        self.fusion = synthes_net()

    def forward(self, nonkey, flow_ini, FlowBackWarp):
        xa = self.conv1a(nonkey)
        xa = F.relu(xa, inplace=True)
        xa = self.conv2a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv3a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv4a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv5a(xa)
        nonkey = nonkey + xa

        flow_fb = self.flownet(nonkey, flow_ini)
        flow_fb = flow_fb + flow_ini
        key_warp = FlowBackWarp(nonkey, flow_fb)
        fusion_frame = self.fusion(key_warp, nonkey)
        return fusion_frame, flow_fb


class basic_block_a_key(nn.Module):
    def __init__(self):
        super(basic_block_a_key, self).__init__()
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.flownet = FLOWNET1_key()
        self.fusion = synthes_net()

    def forward(self, nonkey, FlowBackWarp):
        xa = self.conv1a(nonkey)
        xa = F.relu(xa, inplace=True)
        xa = self.conv2a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv3a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv4a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv5a(xa)
        nonkey = nonkey + xa

        flow_fb = self.flownet(nonkey)
        key_warp = FlowBackWarp(nonkey, flow_fb)
        fusion_frame = self.fusion(key_warp, nonkey)
        return fusion_frame, flow_fb


class sample_and_inirecon_key(nn.Module):
    def __init__(self, num_filters, B_size, device):
        super(sample_and_inirecon_key, self).__init__()
        self.sample = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=B_size, stride=B_size, padding=0,
                                bias=False)

        self.pg_block0 = basic_block_a_key()
        self.pg_block1 = basic_block_b_key()
        self.pg_block2 = basic_block_b_key()
        self.pg_block3 = basic_block_b_key()
        self.pg_block4 = basic_block_b_key()
        self.pg_block5 = basic_block_b_key()
        self.pg_block6 = basic_block_b_key()
        self.pg_block7 = basic_block_b_key()
        self.pg_block8 = basic_block_b_key()

        self.num_filters = num_filters
        self.B_size = B_size
        self.t2image = nn.PixelShuffle(B_size)

        self.FlowBackWarp = backWarp_MH(device)

    def forward(self, x_ini):
        sample_w = self.sample.weight
        sample_w = torch.reshape(sample_w, (self.num_filters, (self.B_size * self.B_size)))
        sample_w_t = sample_w.t()

        sample_w_t = torch.unsqueeze(sample_w_t, 2)
        t_mat = torch.unsqueeze(sample_w_t, 3)

        phi_x = self.sample(x_ini)
        zk = F.conv2d(phi_x, t_mat, stride=1, padding=0)
        zk = self.t2image(zk)
        xoutput, flow_out = self.pg_block0(zk, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block1(zk, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block2(zk, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block3(zk, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block4(zk, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block5(zk, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block6(zk, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block7(zk, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block8(zk, flow_out, self.FlowBackWarp)

        return xoutput


class FLOWNET1_nonkey(nn.Module):
    def __init__(self):
        super(FLOWNET1_nonkey, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of + 1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=64 + 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=num_of * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3):
        # x = torch.cat((x1, x2, x3), dim=1)
        x2 = x2.repeat(1, num_of1, 1, 1)
        x3 = x3.repeat(1, num_of2, 1, 1)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        xa = F.relu(x, inplace=True)
        x = F.interpolate(xa, scale_factor=0.5, mode='bilinear')

        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        xb = F.relu(x, inplace=True)
        x = F.interpolate(xb, scale_factor=0.5, mode='bilinear')

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)

        up1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xb, up1), dim=1)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)

        up2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xa, up2), dim=1)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        output = self.conv10(x)

        return output


class basic_block_a_nonkey(nn.Module):
    def __init__(self):
        super(basic_block_a_nonkey, self).__init__()
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.flownet = FLOWNET1_nonkey()
        self.fusion = synthes_net()

    def forward(self, nonkey, key1, key2, FlowBackWarp):
        xa = self.conv1a(nonkey)
        xa = F.relu(xa, inplace=True)
        xa = self.conv2a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv3a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv4a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv5a(xa)
        nonkey = nonkey + xa

        flow_fb = self.flownet(nonkey, key1, key2)
        key1_warp = FlowBackWarp(key1, flow_fb[:, 0:num_of, :, :], num_of1)
        key2_warp = FlowBackWarp(key2, flow_fb[:, num_of:2 * num_of, :, :], num_of2)
        key_warp = torch.cat((key1_warp, key2_warp), dim=1)
        fusion_frame = self.fusion(key_warp, nonkey)
        return fusion_frame, flow_fb



class FLOWNET2_nonkey(nn.Module):
    def __init__(self):
        super(FLOWNET2_nonkey, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of * 3 + 1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=64 + 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=num_of * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4):
        x2 = x2.repeat(1, num_of1, 1, 1)
        x3 = x3.repeat(1, num_of2, 1, 1)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        xa = F.relu(x, inplace=True)
        x = F.interpolate(xa, scale_factor=0.5, mode='bilinear')

        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        xb = F.relu(x, inplace=True)
        x = F.interpolate(xb, scale_factor=0.5, mode='bilinear')

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)

        up1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xb, up1), dim=1)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)

        up2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xa, up2), dim=1)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        output = self.conv10(x)

        return output


class basic_block_b_nonkey(nn.Module):
    def __init__(self):
        super(basic_block_b_nonkey, self).__init__()
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.flownet = FLOWNET2_nonkey()
        self.fusion = synthes_net()

    def forward(self, nonkey, key1, key2, flow_ini, FlowBackWarp):
        xa = self.conv1a(nonkey)
        xa = F.relu(xa, inplace=True)
        xa = self.conv2a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv3a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv4a(xa)
        xa = F.relu(xa, inplace=True)
        xa = self.conv5a(xa)
        nonkey = nonkey + xa

        flow_fb = self.flownet(nonkey, key1, key2, flow_ini)
        flow_fb = flow_fb + flow_ini
        key1_warp = FlowBackWarp(key1, flow_fb[:, 0:num_of, :, :], num_of1)
        key2_warp = FlowBackWarp(key2, flow_fb[:, num_of:2 * num_of, :, :], num_of2)
        key_warp = torch.cat((key1_warp, key2_warp), dim=1)
        fusion_frame = self.fusion(key_warp, nonkey)
        return fusion_frame, flow_fb



class sample_and_inirecon_nonkey(nn.Module):
    def __init__(self, num_filters, B_size, device):
        super(sample_and_inirecon_nonkey, self).__init__()
        self.sample = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=B_size, stride=B_size, padding=0,
                                bias=False)

        self.pg_block0 = basic_block_a_nonkey()
        self.pg_block1 = basic_block_b_nonkey()
        self.pg_block2 = basic_block_b_nonkey()
        self.pg_block3 = basic_block_b_nonkey()
        self.pg_block4 = basic_block_b_nonkey()
        self.pg_block5 = basic_block_b_nonkey()
        self.pg_block6 = basic_block_b_nonkey()
        self.pg_block7 = basic_block_b_nonkey()
        self.pg_block8 = basic_block_b_nonkey()

        self.num_filters = num_filters
        self.B_size = B_size
        self.t2image = nn.PixelShuffle(B_size)

        self.FlowBackWarp = backWarp_MH(device)

    def forward(self, x_ini, x_key1, x_key2):
        sample_w = self.sample.weight
        sample_w = torch.reshape(sample_w, (self.num_filters, (self.B_size * self.B_size)))
        sample_w_t = sample_w.t()

        sample_w_t = torch.unsqueeze(sample_w_t, 2)
        t_mat = torch.unsqueeze(sample_w_t, 3)

        phi_x = self.sample(x_ini)
        zk = F.conv2d(phi_x, t_mat, stride=1, padding=0)
        zk = self.t2image(zk)
        xoutput, flow_out = self.pg_block0(zk, x_key1, x_key2, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block1(zk, x_key1, x_key2, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block2(zk, x_key1, x_key2, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block3(zk, x_key1, x_key2, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block4(zk, x_key1, x_key2, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block5(zk, x_key1, x_key2, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block6(zk, x_key1, x_key2, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block7(zk, x_key1, x_key2, flow_out, self.FlowBackWarp)

        phi_xoutput = self.sample(xoutput)
        delta_phi = phi_x - phi_xoutput
        delta_xty = F.conv2d(delta_phi, t_mat, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        zk = delta_xty + xoutput
        xoutput, flow_out = self.pg_block8(zk, x_key1, x_key2, flow_out, self.FlowBackWarp)

        return xoutput


