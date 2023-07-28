# Implementation of
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

# Adapted from https://github.com/rcdaudt/fully_convolutional_change_detection/blob/master/siamunet_diff.py

# # Original head information
# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

# Dropout layers are disabled by default

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import Conv3x3, MaxPool2x2, ConvTransposed3x3
from ._utils import constant_init,normal_init, trunc_normal_init, PatchEmbed, nchw_to_nlc, nlc_to_nchw,Identity
from ._common import MultiheadAttention
from .hmcdnet import *



class SiamUNet_EF_tfblock(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, use_dropout=False):
        super().__init__()

        self.use_dropout = use_dropout
        self.layers = self.tfblcok()

        self.conv11 = Conv3x3(in_ch*2, 16, norm=True, act=True)
        self.do11 = self.make_dropout()
        self.conv12 = Conv3x3(16, 16, norm=True, act=True)
        self.do12 = self.make_dropout()
        


        self.upconv4 = ConvTransposed3x3(256, 128, output_padding=1)

        self.conv43d = Conv3x3(128+160, 128, norm=True, act=True)
        self.do43d = self.make_dropout()
        self.conv42d = Conv3x3(128, 128, norm=True, act=True)
        self.do42d = self.make_dropout()
        self.conv41d = Conv3x3(128, 64, norm=True, act=True)
        self.do41d = self.make_dropout()

        self.upconv3 = ConvTransposed3x3(64, 64, output_padding=1)

        self.conv33d = Conv3x3(128, 64, norm=True, act=True)
        self.do33d = self.make_dropout()
        self.conv32d = Conv3x3(64, 64, norm=True, act=True)
        self.do32d = self.make_dropout()
        self.conv31d = Conv3x3(64, 32, norm=True, act=True)
        self.do31d = self.make_dropout()

        self.upconv2 = ConvTransposed3x3(32, 32, output_padding=1)

        self.conv22d = Conv3x3(64, 32, norm=True, act=True)
        self.do22d = self.make_dropout()
        self.conv21d = Conv3x3(32, 16, norm=True, act=True)
        self.do21d = self.make_dropout()

        self.upconv1 = ConvTransposed3x3(16, 16, output_padding=1)

        self.conv12d = Conv3x3(32, 16, norm=True, act=True)
        self.do12d = self.make_dropout()
        self.conv11d = Conv3x3(16, out_ch)

        self.conv12d_3d = Conv3x3(32, 16, norm=True, act=True)
        self.do12d_3d = self.make_dropout()
        self.conv11d_3d = Conv3x3(16, 1)
        self.activate3d = nn.Tanh()

    def tfblcok(self, in_channels=16, num_layers=[2, 2, 2, 2],patch_sizes=[3,3,3,3],strides=[2,2,2,2],embed_dims=32,
                num_heads=[1, 2, 5, 8],norm_cfg=dict(type='LN', eps=1e-6),sr_ratios=[8, 4, 2, 1],
               mlp_ratio=4,qkv_bias=True,drop_rate=0.,attn_drop_rate=0., drop_path_rate=0.1,act_cfg=dict(type='GELU')):
        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule
        cur = 0
        layers = nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                pad_to_patch_size=False,
                norm_cfg=norm_cfg)
            layer = nn.ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i

            norm = nn.LayerNorm(embed_dims_i,eps=1e-6)
            #import pdb;pdb.set_trace()
            layers.append(nn.ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        return layers

    def forward(self, t1, t2):
        # Encode 1
        # Stage 1
        x11 = self.do11(self.conv11(torch.cat([t1,t2],1)))
        x12_1 = self.do12(self.conv12(x11))
        
        
        
        x1p, H, W = self.layers[0][0](x12_1), self.layers[0][0].DH, self.layers[0][0].DW
        hw_shape = (H, W)
        for block in self.layers[0][1]:
            x1p = block(x1p, hw_shape)
        x1p = self.layers[0][2](x1p)
        x1p = nlc_to_nchw(x1p, hw_shape)

        # Stage 2

        x2p, H, W = self.layers[1][0](x1p), self.layers[1][0].DH, self.layers[1][0].DW
        hw_shape = (H, W)
        for block in self.layers[1][1]:
            x2p = block(x2p, hw_shape)
        x2p = self.layers[1][2](x2p)
        x2p = nlc_to_nchw(x2p, hw_shape)        

        # Stage 3
        x3p, H, W = self.layers[2][0](x2p), self.layers[2][0].DH, self.layers[2][0].DW
        hw_shape = (H, W)
        for block in self.layers[2][1]:
            x3p = block(x3p, hw_shape)
        x3p = self.layers[2][2](x3p)
        x3p = nlc_to_nchw(x3p, hw_shape)   
        
        # Stage 4
        x4p, H, W = self.layers[3][0](x3p), self.layers[3][0].DH, self.layers[3][0].DW
        hw_shape = (H, W)
        for block in self.layers[3][1]:
            x4p = block(x4p, hw_shape)
        x4p = self.layers[3][2](x4p)
        x4p = nlc_to_nchw(x4p, hw_shape)   

        
        # Decode
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = (0, x3p.shape[3] - x4d.shape[3], 0, x3p.shape[2] - x4d.shape[2])
        x4d = torch.cat([F.pad(x4d, pad=pad4, mode='replicate'), x3p], 1)
        x43d = self.do43d(self.conv43d(x4d))
        x42d = self.do42d(self.conv42d(x43d))
        x41d = self.do41d(self.conv41d(x42d))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = (0, x2p.shape[3] - x3d.shape[3], 0, x2p.shape[2] - x3d.shape[2])
        x3d = torch.cat([F.pad(x3d, pad=pad3, mode='replicate'), x2p], 1)
        x33d = self.do33d(self.conv33d(x3d))
        x32d = self.do32d(self.conv32d(x33d))
        x31d = self.do31d(self.conv31d(x32d))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = (0, x1p.shape[3] - x2d.shape[3], 0, x1p.shape[2] - x2d.shape[2])
        x2d = torch.cat([F.pad(x2d, pad=pad2, mode='replicate'), x1p], 1)
        x22d = self.do22d(self.conv22d(x2d))
        x21d = self.do21d(self.conv21d(x22d))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = (0, x12_1.shape[3] - x1d.shape[3], 0, x12_1.shape[2] - x1d.shape[2])
        x1d = torch.cat([F.pad(x1d, pad=pad1, mode='replicate'), x12_1], 1)

        x12d = self.do12d(self.conv12d(x1d))
        x11d = self.conv11d(x12d)

        x12d_3d = self.do12d_3d(self.conv12d_3d(x1d))
        x11d_3d = self.conv11d_3d(x12d_3d)

        x11d_3d = self.activate3d(x11d_3d)
        
        #import pdb;pdb.set_trace()
        return x11d, x11d_3d

    def make_dropout(self):
        if self.use_dropout:
            return nn.Dropout2d(p=0.2)
        else:
            return Identity()
