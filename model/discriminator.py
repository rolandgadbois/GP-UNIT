import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
import math
import numpy as np
from model.base_network import BaseNetwork

class ResnetBlock(BaseNetwork):
    def __init__(self, fin, fout, normalize=False, downsample=False):
        super(ResnetBlock, self).__init__()
        self.learned_shortcut = (fin != fout)
        self.normalize = normalize
        self.downsample = downsample
        fmiddle = fin

        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, 3, 1, 1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, 3, 1, 1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, 1, 1, 0, bias=False))

        if self.normalize:
            self.norm_0 = nn.InstanceNorm2d(fin, affine=True)
            self.norm_1 = nn.InstanceNorm2d(fmiddle, affine=True)

    def forward(self, x):
        x_s = self.shortcut(x)

        if self.normalize:
            x = self.norm_0(x)
        dx = self.conv_0(self.actvn(x))
        if self.downsample:
            dx = F.avg_pool2d(dx, 2)
        if self.normalize:
            dx = self.norm_1(dx)
        dx = self.conv_1(self.actvn(dx))

        out = x_s + dx

        return out / math.sqrt(2)

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        if self.downsample:
            x_s = F.avg_pool2d(x_s, 2)
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class Discriminator(BaseNetwork):
    def __init__(self, max_conv_dim=512):
        super().__init__()
        dim_in = 64
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(3, dim_in, 3, 1, 1))]

        repeat_num = int(np.log2(256)) - 2  # 6 times for 256x256 input
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResnetBlock(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 4, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        self.model = nn.Sequential(*blocks)
        self.linear = spectral_norm(nn.Conv2d(dim_out, 1, 1, 1, 0))

    def forward(self, x):
        feat = self.model(x)
        out = self.linear(feat)
        return out.view(-1)
