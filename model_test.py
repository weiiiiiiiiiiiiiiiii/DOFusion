import torch
import numpy as np
import torch.nn as nn

import torch
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import kornia
from PIL import Image
import math
from torch.autograd import Variable


from sobel import ContrastEnhance
from utils import compute_mask
from utils import compute_mask_ir_vis
# import django
####################################################################################################################################
from utils import compute_w


##########this  is Atrous Convolutional################################################################################################
class DenseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseConv, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.dense_conv = conv(in_channels, out_channels)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


class DenseBlock1(nn.Module):
    def __init__(self):
        super(DenseBlock1, self).__init__()
        denseblock = []
        denseblock += [DenseConv(16, 16),
                       DenseConv(32, 16),
                       DenseConv(48, 16)]
        self.denseblock = nn.Sequential(*denseblock)
        self.denseblock_tail = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())

    def forward(self, x):
        out = self.denseblock(x)
        out = self.denseblock_tail(out)
        # out = torch.cat([x, out], 1)
        return out



# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def conv(in_channels, out_channels, kernel_size=3, bias=False, stride=1):
    return nn.Sequential(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride),nn.ReLU())


##########################################################################################################

###########################################################################################################
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.ReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class ResidualFlow(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(ResidualFlow, self).__init__()

        self.group_conv1 = nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, bias=bias, groups=groups)
        self.group_conv2 = nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, bias=bias, groups=groups)
        self.group_conv3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias, groups=groups)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat // 2, n_feat // 2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      bias=bias, groups=groups),
            nn.BatchNorm2d(n_feat // 2),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // 2, n_feat // 2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      bias=bias, groups=groups),
            nn.BatchNorm2d(n_feat // 2)
        )



        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        group1 = self.group_conv1(x)
        group2 = self.group_conv2(x)

        out = torch.cat((group1, self.body(group2)), dim=1)
        out = self.group_conv3(out)

        out += residual
        out = self.act(out)

        return out




class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x




class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        x = self.bot(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

#######################################################################################

##########################################################################################################

################################################################################################################
##---------- Multi-Scale Resiudal Block (MRB) ----------
class MRB(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias):
        super(MRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width


        self.dau_top = ResidualFlow(int(n_feat * (chan_factor + 0.5) ** -2), bias=bias)
        self.dau_mid = ResidualFlow(int(n_feat * (chan_factor + 0.5) ** -1), bias=bias)
        self.dau_bot = ResidualFlow(int(n_feat * (chan_factor + 0.5) ** 0), bias=bias)

        self.up2 = UpSample(int(((chan_factor + 0.5) ** 0) * n_feat), 2, (chan_factor + 0.5))
        self.up4 = nn.Sequential(
            UpSample(int(((chan_factor + 0.5) ** 0) * n_feat), 2, (chan_factor + 0.5)),
            UpSample(int(((chan_factor + 0.5) ** -1) * n_feat), 2, (chan_factor + 0.5))
        )

        self.down21_1 = DownSample(int(((chan_factor + 0.5) ** -1) * n_feat), 2, (chan_factor + 0.5))
        self.down21_2 = DownSample(int(((chan_factor + 0.5) ** -1) * n_feat), 2, (chan_factor + 0.5))
        self.down32_1 = DownSample(int(((chan_factor + 0.5) ** -2) * n_feat), 2, (chan_factor + 0.5))
        self.down32_2 = DownSample(int(((chan_factor + 0.5) ** -2) * n_feat), 2, (chan_factor + 0.5))

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        self.skff_bot = SKFF(int(n_feat * (chan_factor + 0.5) ** 0), 2)
        self.skff_bot_f = SKFF(int(n_feat * (chan_factor + 0.5) ** 0), 3)
        self.skff_mid = SKFF(int(n_feat * (chan_factor + 0.5) ** -1), 2)

        # self.ACE = ACE()

    def forward(self, x, ):
        x_bot = x.clone()
        x_mid = self.up2(x_bot)

        x_mid1 = self.dau_mid(x_mid)
        x_bot1 = self.dau_bot(x_bot)

        x_mid2 = self.down21_1(x_mid)
        x_bot2 = self.skff_bot([x_bot1, x_mid2])

        x_bot3 = self.dau_bot(x_bot2)
        x_mid3 = self.dau_mid(x_mid1)

        x_mid5 = self.down21_2(x_mid3)
        x_mid4 = self.skff_bot([x_bot3, x_mid5])

        out = self.dau_bot(x_mid4)

        out = out + x

        return out



# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXfuse stage######################################################
class Conv(nn.Module):
    def __init__(self, out_ch):
        super(Conv, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.LeakyReLU(negative_slope=1e-2, inplace=True))

    def forward(self, x):
        return self.body(x)


class UDIF_net(nn.Module):
    def __init__(self, args):
        super(UDIF_net, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.ReLU())

        self.ds1 = DenseBlock1()

        self.d_top = kornia.filters.BoxBlur((8, 8))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ace = ContrastEnhance().to(device)
        module_msrb1 = [MRB(64, chan_factor=1.5, height=3, width=2, bias=False)]


        self.msrb1 = nn.Sequential(*module_msrb1)

        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.ReLU())


        self.restruction1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU())
        self.restruction2 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU())
        self.restruction3 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=True),
                                          )

        self.sig = nn.Sigmoid()


    def forward(self, x1, x2):
        F1 = self.conv1(x1)
        F2 = self.conv1(x2)

        F1 = self.ds1(F1)
        F2 = self.ds1(F2)
        #
        F3 = F1 + F2
        #########################

        F1_D = F1 - self.d_top(F1)
        F2_D = F2 - self.d_top(F2)

        F11_D = self.ace(F1_D)
        F22_D = self.ace(F2_D)


        ########################################
        F_11 = self.msrb1(F11_D)
        F_22 = self.msrb1(F22_D)
        #F_11 = self.msrb1(F1)
        #F_22 = self.msrb1(F2)
        #F_11 = self.conv6(F11_D)
        #F_22 = self.conv6(F22_D)
        F = F_11 + F_22
        #F = torch.clamp(F, 0, 1)
        F_out = F3 + F




        output = self.restruction1(F_out)
        output = self.restruction2(output)

        output = self.restruction3(output)


        output = self.sig(output)


        return output


