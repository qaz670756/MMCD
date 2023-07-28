# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from ..builder import BACKBONES
import torch
from .cgnet import ContextGuidedBlock,InputInjection
import torch.nn.functional as F
from mmseg.ops import resize

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output
class up(nn.Module):
    def __init__(self, in_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x
def cgblock(in_ch, out_ch, dilation=2, reduction=8, conv2d=False):
    norm_cfg = {'type': 'SyncBN', 'eps': 0.001, 'requires_grad': True}
    act_cfg = {'type': 'PReLU', 'num_parameters': 32}
    if not conv2d:
        return ContextGuidedBlock(in_ch, out_ch,
                           dilation=dilation,
                           reduction=reduction,
                           downsample=False,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg,skip_connect=False)
    else:
        return conv_block_nested(in_ch, out_ch, out_ch)
@BACKBONES.register_module()
class SiameseCGNet(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, bilinear=True,diffFPN=False,conv2d=False,mid_cam=False):
        super(SiameseCGNet, self).__init__()
        torch.nn.Module.dump_patches = True

        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        dilations = (1, 2, 4, 8, 16)
        reductions = (4, 8, 16, 32, 64)

        self.diffFPN = diffFPN
        self.mid_cam=mid_cam

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up1_0 = up(filters[1],bilinear)
        self.Up2_0 = up(filters[2],bilinear)
        self.Up3_0 = up(filters[3],bilinear)
        self.Up4_0 = up(filters[4],bilinear)



        if not self.diffFPN:
            self.conv0_0 = cgblock(in_ch, filters[0],conv2d=conv2d)
            self.conv1_0 = cgblock(filters[0], filters[1],conv2d=conv2d)
            self.conv2_0 = cgblock(filters[1], filters[2],conv2d=conv2d)
            self.conv3_0 = cgblock(filters[2], filters[3],conv2d=conv2d)
            self.conv4_0 = cgblock(filters[3], filters[4],conv2d=conv2d)

            self.conv0_1 = cgblock(filters[0] * 2 + filters[1], filters[0],conv2d=conv2d)
            self.conv1_1 = cgblock(filters[1] * 2 + filters[2], filters[1],conv2d=conv2d)
            self.Up1_1 = up(filters[1],bilinear)
            self.conv2_1 = cgblock(filters[2] * 2 + filters[3], filters[2],conv2d=conv2d)
            self.Up2_1 = up(filters[2],bilinear)
            self.conv3_1 = cgblock(filters[3] * 2 + filters[4], filters[3],conv2d=conv2d)
            self.Up3_1 = up(filters[3],bilinear)

            self.conv0_2 = cgblock(filters[0] * 3 + filters[1], filters[0],conv2d=conv2d)
            self.conv1_2 = cgblock(filters[1] * 3 + filters[2], filters[1],conv2d=conv2d)
            self.Up1_2 = up(filters[1],bilinear)
            self.conv2_2 = cgblock(filters[2] * 3 + filters[3], filters[2],conv2d=conv2d)
            self.Up2_2 = up(filters[2],bilinear)

            self.conv0_3 = cgblock(filters[0] * 4 + filters[1], filters[0],conv2d=conv2d)
            self.conv1_3 = cgblock(filters[1] * 4 + filters[2], filters[1],conv2d=conv2d)
            self.Up1_3 = up(filters[1],bilinear)

            self.conv0_4 = cgblock(filters[0] * 5 + filters[1], filters[0],conv2d=conv2d)
        else:
            self.conv0_0 = cgblock(in_ch, filters[0], dilations[0], reductions[0],conv2d=conv2d)
            self.conv1_0 = cgblock(filters[0], filters[1], dilations[1], reductions[1],conv2d=conv2d)
            self.conv2_0 = cgblock(filters[1], filters[2], dilations[2], reductions[2],conv2d=conv2d)
            self.conv3_0 = cgblock(filters[2], filters[3], dilations[3], reductions[3],conv2d=conv2d)
            self.conv4_0 = cgblock(filters[3], filters[4], dilations[4], reductions[4],conv2d=conv2d)
            pairs = len(filters)
            # lateral convs for unifing channels
            self.lateral_convs = nn.ModuleList()
            for i in range(pairs):
                self.lateral_convs.append(
                    cgblock(filters[i]*2, filters[i], dilations[i], reductions[i],conv2d=conv2d)
                )
            # top_down_convs
            self.top_down_convs = nn.ModuleList()
            for i in range(pairs-1):
                self.top_down_convs.append(
                    cgblock(filters[i+1], filters[i], dilation=dilations[i+1], reduction=reductions[i+1],conv2d=conv2d))

            # diff convs
            self.diff_convs_col1 = nn.ModuleList()
            for i in range(pairs-1):
                self.diff_convs_col1.append(
                    cgblock(filters[i]*3, filters[i], dilations[i], reductions[i],conv2d=conv2d))

            self.diff_convs_col2 = nn.ModuleList()
            for i in range(pairs-2):
                self.diff_convs_col2.append(
                    cgblock(filters[i]*3, filters[i], dilations[i], reductions[i],conv2d=conv2d))

            self.diff_convs_col3 = nn.ModuleList()
            for i in range(pairs - 3):
                self.diff_convs_col3.append(
                    cgblock(filters[i]*3, filters[i],
                            dilation=dilations[i], reduction=reductions[i],conv2d=conv2d))

            self.diff_convs_col4 = cgblock(filters[0]*3, filters[0],
                        dilation=dilations[0], reduction=reductions[0],conv2d=conv2d)

            self.up2x = up(32, bilinear)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_inputs(self, inputs):
        outputs = [
            resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=False) for x in inputs
        ]
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def various_distance(self, feature_pairs):
        fea1, fea2 = feature_pairs
        n, c, h, w = fea1.shape
        fea1_rz = torch.transpose(fea1.view(n, c, h * w), 2, 1)
        fea2_rz = torch.transpose(fea2.view(n, c, h * w), 2, 1)
        return F.pairwise_distance(fea1_rz,fea2_rz,p=2)

    def forward(self, x):
        '''xA'''
        bsize = x.shape[0]//2
        #x = torch.cat([xA,xB],dim=0)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x0_0A, x0_0B = x0_0[:bsize,::], x0_0[bsize:,::]
        x1_0A, x1_0B = x1_0[:bsize,::], x1_0[bsize:,::]
        x2_0A, x2_0B = x2_0[:bsize,::], x2_0[bsize:,::]
        x3_0A, x3_0B = x3_0[:bsize,::], x3_0[bsize:,::]
        x4_0A, x4_0B = x4_0[:bsize,::], x4_0[bsize:,::]

        output = [x0_0A, x0_0B, x1_0A, x1_0B, x2_0A, x2_0B, x3_0A, x3_0B, x4_0A, x4_0B]

        mid = self._transform_inputs((x0_0, x1_0, x2_0, x3_0))
        n, c, h, w = mid.shape
        n = n // 2
        mid = [mid[:n, ::], mid[n:, ::]]
        distance = self.various_distance(mid).view(n, 1, h, w)

        if not self.diffFPN:

            x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
            x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

            x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

            x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        elif self.mid_cam:

            pairs = len(output)//2
            tmp = [self.lateral_convs[i](torch.cat([output[i * 2], output[i * 2 + 1]], dim=1)
                            + F.interpolate(distance, size=output[i * 2].shape[2:],
                              align_corners=False, mode='bilinear'))
                   for i in range(pairs)]

            # top_down_path
            for i in range(pairs-1, 0, -1):
                tmp[i - 1] += self.up2x(self.top_down_convs[i-1](tmp[i]))

            # x0_1
            tmp = [self.diff_convs_col1[i](torch.cat([tmp[i], self.up2x(tmp[i + 1])], dim=1))
                   for i in range(pairs-1)]
            x0_1 = tmp[0]
            # x0_2
            tmp = [self.diff_convs_col2[i](torch.cat([tmp[i], self.up2x(tmp[i+1])], dim=1))
                   for i in range(pairs-2)]
            x0_2 = tmp[0]
            # x0_3
            tmp = [self.diff_convs_col3[i](torch.cat([tmp[i], self.up2x(tmp[i+1])], dim=1))
                                           for i in range(pairs-3)]
            x0_3 = tmp[0]
            # x0_4
            x0_4 = self.diff_convs_col4(torch.cat([tmp[0], self.up2x(tmp[1])], dim=1))
        else:

            pairs = len(output)//2
            tmp = [self.lateral_convs[i](torch.cat([output[i * 2], output[i * 2 + 1]], dim=1))
                   for i in range(pairs)]

            # top_down_path
            for i in range(pairs-1, 0, -1):
                tmp[i - 1] += self.up2x(self.top_down_convs[i-1](tmp[i]))

            # x0_1
            tmp = [self.diff_convs_col1[i](torch.cat([tmp[i], self.up2x(tmp[i + 1])], dim=1))
                   for i in range(pairs-1)]
            x0_1 = tmp[0]
            # x0_2
            tmp = [self.diff_convs_col2[i](torch.cat([tmp[i], self.up2x(tmp[i+1])], dim=1))
                   for i in range(pairs-2)]
            x0_2 = tmp[0]
            # x0_3
            tmp = [self.diff_convs_col3[i](torch.cat([tmp[i], self.up2x(tmp[i+1])], dim=1))
                                           for i in range(pairs-3)]
            x0_3 = tmp[0]
            # x0_4
            x0_4 = self.diff_convs_col4(torch.cat([tmp[0], self.up2x(tmp[1])], dim=1))


        x = self._transform_inputs((x0_1, x0_2, x0_3, x0_4))
        return x, mid, distance


