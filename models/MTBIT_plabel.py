import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models
import math
from timm.models.layers import trunc_normal_
import functools
from einops import rearrange
import math

import models.resnet as rn
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d


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
    def __init__(self, in_features=1, hidden_features=32, out_features=1, act_layer=nn.GELU, drop=0.1):
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

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.sigmoid(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return x


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18', par_HE=False,
                 output_sigmoid=False, if_upsample_2x=True, learnable=False, sigmoid3d=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = rn.resnet18(pretrained=True,
                                      replace_stride_with_dilation=[False, True, True])

        elif backbone == 'resnet34':
            self.resnet = rn.resnet34(pretrained=True,
                                      replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = rn.resnet50(pretrained=True,
                                      replace_stride_with_dilation=[False, True, True])
            expand = 4
        elif backbone == 'dino:resnet50':
            self.resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            expand = 4
        elif backbone == 'encoder:resnet50':
            self.resnet = models.resnet50(pretrained=True)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()

        self.upsamplex2 = nn.Upsample(scale_factor=2)

        self.learnable = learnable

        self.upsamplex2l1_single = nn.ConvTranspose2d(256 * expand, 256 * expand, 4, 2, 1)

        if self.learnable:
            self.upsamplex2l1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
            self.upsamplex2l2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        # For ensuring the reproducility
        # refer to https://github.com/pytorch/pytorch/issues/7068#issuecomment-716719820
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.regressor = TwoLayerConv2d(in_channels=32, out_channels=1)
        self.soft_thresh = SoftThresh(out_features=1)

        self.classifier_plabel = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        if par_HE:
            self.regressor_HE = TwoLayerConv2d(in_channels=32, out_channels=1)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        if sigmoid3d == True:
            self.active3d = nn.Sigmoid()
        else:
            self.active3d = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1_single(x_8)
            else:
                x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class MTBIT_plabel(ResNet):
    def __init__(self, input_nc=3, output_nc=3, resnet_stages_num=4,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=8,
                 dim_head=64, decoder_dim_head=16,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable=False,
                 decoder_softmax=True,
                 ):

        super(MTBIT_plabel, self).__init__(input_nc, output_nc, backbone=backbone,
                                                      resnet_stages_num=resnet_stages_num,
                                                      if_upsample_2x=if_upsample_2x,
                                                      learnable=learnable, )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)
        self.learnable = learnable
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2 * dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))
        decoder_pos_size = 1024 // 4
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
                                                              decoder_pos_size,
                                                              decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                                                      heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward_features(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)
        # feature differencing
        x = x2 - x1

        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)

        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)

        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)
        x3d = self.active3d(x3d)

        # import pdb;pdb.set_trace()
        thresh_3d = self.soft_thresh(x3d)
        xplabel = self.classifier_plabel(x + thresh_3d)

        return x3d, x2d, xplabel  # ,thresh_3d

    def forward(self, x1=None, x2=None):
        vis_flag = 0
        if x1 is None or x2 is None:
            vis_flag = 1
            bsize = x1.shape[0] // 2
            x1, x2 = x1[:bsize, :, :, :].requires_grad_(), x1[bsize:, :, :, :].requires_grad_()

        if not vis_flag:
            return self.forward_features(x1, x2)
        else:
            # return the 2d prediction for feature visualization
            return [self.forward_features(x1, x2)[1]]



