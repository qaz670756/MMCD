# Implementation of
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

# Adapted from https://github.com/rcdaudt/fully_convolutional_change_detection/blob/master/siamunet_conc.py

# # Original head information
# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

# Dropout layers are disabled by default
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import  trunc_normal_
from ._utils._blocks import Conv3x3, MaxPool2x2, ConvTransposed3x3
from ._utils._utils import Identity


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=16)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class SoftThresh(nn.Module):
    def __init__(self, in_features=48, hidden_features=64, out_features=1, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        B,C,H,W = x.shape
        x = self.sigmoid(x)
        x = x*y
        x = x.flatten(2).transpose(1, 2)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0,2,1).reshape(B, -1, H, W)
        return x
    
class siamunet_conc_plabel(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, use_dropout=False):
        super().__init__()

        self.use_dropout = use_dropout

        self.conv11 = Conv3x3(in_ch, 16, norm=True, act=True)
        self.do11 = self.make_dropout()
        self.conv12 = Conv3x3(16, 16, norm=True, act=True)
        self.do12 = self.make_dropout()
        self.pool1 = MaxPool2x2()

        self.conv21 = Conv3x3(16, 32, norm=True, act=True)
        self.do21 = self.make_dropout()
        self.conv22 = Conv3x3(32, 32, norm=True, act=True)
        self.do22 = self.make_dropout()
        self.pool2 = MaxPool2x2()

        self.conv31 = Conv3x3(32, 64, norm=True, act=True)
        self.do31 = self.make_dropout()
        self.conv32 = Conv3x3(64, 64, norm=True, act=True)
        self.do32 = self.make_dropout()
        self.conv33 = Conv3x3(64, 64, norm=True, act=True)
        self.do33 = self.make_dropout()
        self.pool3 = MaxPool2x2()

        self.conv41 = Conv3x3(64, 128, norm=True, act=True)
        self.do41 = self.make_dropout()
        self.conv42 = Conv3x3(128, 128, norm=True, act=True)
        self.do42 = self.make_dropout()
        self.conv43 = Conv3x3(128, 128, norm=True, act=True)
        self.do43 = self.make_dropout()
        self.pool4 = MaxPool2x2()

        self.upconv4 = ConvTransposed3x3(128, 128, output_padding=1)

        self.conv43d = Conv3x3(384, 128, norm=True, act=True)
        self.do43d = self.make_dropout()
        self.conv42d = Conv3x3(128, 128, norm=True, act=True)
        self.do42d = self.make_dropout()
        self.conv41d = Conv3x3(128, 64, norm=True, act=True)
        self.do41d = self.make_dropout()

        self.upconv3 = ConvTransposed3x3(64, 64, output_padding=1)

        self.conv33d = Conv3x3(192, 64, norm=True, act=True)
        self.do33d = self.make_dropout()
        self.conv32d = Conv3x3(64, 64, norm=True, act=True)
        self.do32d = self.make_dropout()
        self.conv31d = Conv3x3(64, 32, norm=True, act=True)
        self.do31d = self.make_dropout()

        self.upconv2 = ConvTransposed3x3(32, 32, output_padding=1)

        self.conv22d = Conv3x3(96, 32, norm=True, act=True)
        self.do22d = self.make_dropout()
        self.conv21d = Conv3x3(32, 16, norm=True, act=True)
        self.do21d = self.make_dropout()

        self.upconv1 = ConvTransposed3x3(16, 16, output_padding=1)

        self.conv12d = Conv3x3(48, 16, norm=True, act=True)
        self.do12d = self.make_dropout()
        self.conv11d = Conv3x3(16, out_ch)
        
        self.soft_thresh = SoftThresh(out_features=48)
        self.plabel_conv = nn.Sequential(Conv3x3(48, 16, norm=True, act=True),
                                         self.make_dropout(),
                                         Conv3x3(16, out_ch))
        
        self.conv12d_3d = Conv3x3(48, 16, norm=True, act=True)
        self.do12d_3d = self.make_dropout()
        self.conv11d_3d = Conv3x3(16, 1)
        self.activate3d = nn.Tanh()

    def forward(self, t1, t2):
        # Encode 1
        # Stage 1
        x11 = self.do11(self.conv11(t1))
        x12_1 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12_1)

        # Stage 2
        x21 = self.do21(self.conv21(x1p))
        x22_1 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22_1)

        # Stage 3
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33_1 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33_1)

        # Stage 4
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43_1 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43_1)

        # Encode 2
        # Stage 1
        x11 = self.do11(self.conv11(t2))
        x12_2 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12_2)

        # Stage 2
        x21 = self.do21(self.conv21(x1p))
        x22_2 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22_2)

        # Stage 3
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33_2 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33_2)

        # Stage 4
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43_2 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43_2)
        
        # Decode
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = (0, x43_1.shape[3]-x4d.shape[3], 0, x43_1.shape[2]-x4d.shape[2])
        x4d = torch.cat([F.pad(x4d, pad=pad4, mode='replicate'), x43_1, x43_2], 1)
        x43d = self.do43d(self.conv43d(x4d))
        x42d = self.do42d(self.conv42d(x43d))
        x41d = self.do41d(self.conv41d(x42d))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = (0, x33_1.shape[3]-x3d.shape[3], 0, x33_1.shape[2]-x3d.shape[2])
        x3d = torch.cat([F.pad(x3d, pad=pad3, mode='replicate'), x33_1, x33_2], 1)
        x33d = self.do33d(self.conv33d(x3d))
        x32d = self.do32d(self.conv32d(x33d))
        x31d = self.do31d(self.conv31d(x32d))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = (0, x22_1.shape[3]-x2d.shape[3], 0, x22_1.shape[2]-x2d.shape[2])
        x2d = torch.cat([F.pad(x2d, pad=pad2, mode='replicate'), x22_1, x22_2], 1)
        x22d = self.do22d(self.conv22d(x2d))
        x21d = self.do21d(self.conv21d(x22d))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = (0, x12_1.shape[3]-x1d.shape[3], 0, x12_1.shape[2]-x1d.shape[2])
        x1d = torch.cat([F.pad(x1d, pad=pad1, mode='replicate'), x12_1, x12_2], 1)
        
        
        x12d = self.do12d(self.conv12d(x1d))
        x2d = self.conv11d(x12d)
        
        
        
        
        x12d_3d = self.do12d_3d(self.conv12d_3d(x1d))
        x11d_3d = self.conv11d_3d(x12d_3d)
        
        x3d = self.activate3d(x11d_3d)
        #import pdb;pdb.set_trace()
        thresh_3d = self.soft_thresh(x3d,x1d)
        x2d_plabel = self.plabel_conv(x1d+thresh_3d)

        return x3d, x2d, x2d_plabel

    def make_dropout(self):
        if self.use_dropout:
            return nn.Dropout2d(p=0.2)
        else:
            return Identity()
