# Implementation of
# C. Zhang et al., “A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images,” 2020, doi: 10.1016/J.ISPRSJPRS.2020.06.003.

# Adapted from https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/blob/master/pytorch%20version/DSIFN.py

# # Original head information
# credits: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images

# Dropout layers are disabled by default

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from ._blocks import Conv1x1, make_norm
from ._common import ChannelAttention, SpatialAttention


class VGG16FeaturePicker(nn.Module):
    def __init__(self, indices=(3,8,15,22,29)):
        super().__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.ModuleList(features).eval()
        self.indices = set(indices)

    def forward(self, x):
        picked_feats = []
        for idx, model in enumerate(self.features):
            x = model(x)
            if idx in self.indices:
                picked_feats.append(x)
        return picked_feats


def conv2d_bn(in_channels, out_ch, with_dropout=True):
    lst = [
        nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        make_norm(out_ch),
    ]
    if with_dropout:
        lst.append(nn.Dropout(p=0.6))
    return nn.Sequential(*lst)


class DSIFN_DEEP3D(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()

        self.encoder1 = self.encoder2 = VGG16FeaturePicker()

        self.sa1 = SpatialAttention()
        self.sa2= SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.ca1 = ChannelAttention(in_channels=1024)
        self.bn_ca1 = make_norm(1024)
        self.o1_conv1 = conv2d_bn(1024, 512, use_dropout)
        self.o1_conv2 = conv2d_bn(512, 512, use_dropout)
        self.bn_sa1 = make_norm(512)
        self.o1_conv3 = Conv1x1(512, 2)
        self.o1_conv3_3d = Conv1x1(512, 1)
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.ca2 = ChannelAttention(in_channels=1536)
        self.bn_ca2 = make_norm(1536)
        self.o2_conv1 = conv2d_bn(1536, 512, use_dropout)
        self.o2_conv2 = conv2d_bn(512, 256, use_dropout)
        self.o2_conv3 = conv2d_bn(256, 256, use_dropout)
        self.bn_sa2 = make_norm(256)
        self.o2_conv4 = Conv1x1(256, 2)
        self.o2_conv4_3d = Conv1x1(256, 1)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.ca3 = ChannelAttention(in_channels=768)
        self.o3_conv1 = conv2d_bn(768, 256, use_dropout)
        self.o3_conv2 = conv2d_bn(256, 128, use_dropout)
        self.o3_conv3 = conv2d_bn(128, 128, use_dropout)
        self.bn_sa3 = make_norm(128)
        self.o3_conv4 = Conv1x1(128, 2)
        self.o3_conv4_3d = Conv1x1(128, 1)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.ca4 = ChannelAttention(in_channels=384)
        self.o4_conv1 = conv2d_bn(384, 128, use_dropout)
        self.o4_conv2 = conv2d_bn(128, 64, use_dropout)
        self.o4_conv3 = conv2d_bn(64, 64, use_dropout)
        self.bn_sa4 = make_norm(64)
        self.o4_conv4 = Conv1x1(64, 2)
        self.o4_conv4_3d = Conv1x1(64, 1)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.ca5 = ChannelAttention(in_channels=192)
        self.o5_conv1 = conv2d_bn(192, 64, use_dropout)
        self.o5_conv2 = conv2d_bn(64, 32, use_dropout)
        self.o5_conv3 = conv2d_bn(32, 16, use_dropout)
        self.bn_sa5 = make_norm(16)
        self.o5_conv4 = Conv1x1(16, 2)
        self.o5_conv4_3d = Conv1x1(16, 1)
        self.activate_3d = nn.Tanh()

    def forward(self, t1, t2):
        # Extract bi-temporal features
        with torch.no_grad():
            self.encoder1.eval(), self.encoder2.eval()
            t1_feats = self.encoder1(t1)
            t2_feats = self.encoder2(t2)
        
        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = t1_feats
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29,= t2_feats

        # Multi-level decoding
        x = torch.cat([t1_f_l29, t2_f_l29], dim=1)
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        out1 = self.o1_conv3(x)
        out1_3d = self.o1_conv3_3d(x)
        out1_3d = self.activate_3d(out1_3d)

        x = self.trans_conv1(x)
        x = torch.cat([x, t1_f_l22, t2_f_l22], dim=1)
        x = self.ca2(x)*x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) *x
        x = self.bn_sa2(x)

        out2 = self.o2_conv4(x)
        out2_3d = self.o2_conv4_3d(x)
        out2_3d = self.activate_3d(out2_3d)

        x = self.trans_conv2(x)
        x = torch.cat([x, t1_f_l15, t2_f_l15], dim=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) *x
        x = self.bn_sa3(x)

        out3 = self.o3_conv4(x)
        out3_3d = self.o3_conv4_3d(x)
        out3_3d = self.activate_3d(out3_3d)

        x = self.trans_conv3(x)
        x = torch.cat([x, t1_f_l8, t2_f_l8], dim=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) *x
        x = self.bn_sa4(x)

        out4 = self.o4_conv4(x)
        out4_3d = self.o4_conv4_3d(x)
        out4_3d = self.activate_3d(out4_3d)
        

        x = self.trans_conv4(x)
        x = torch.cat([x, t1_f_l3, t2_f_l3], dim=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) *x
        x = self.bn_sa5(x)


        out5 = self.o5_conv4(x)
        out5_3d = self.o5_conv4_3d(x)
        out5_3d = self.activate_3d(out5_3d)
        
        #import pdb;pdb.set_trace()

        return [out5, out4, out3, out2, out1], [out5_3d, out4_3d, out3_3d, out2_3d, out1_3d]
