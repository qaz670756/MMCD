import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models 

import functools
from einops import rearrange

import models.resnet as rn
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18', par_HE = False,
                 output_sigmoid=False, if_upsample_2x=True, learnable = False, sigmoid3d=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = rn.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
           
        elif backbone == 'resnet34':
            self.resnet = rn.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = rn.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        elif backbone == 'dino:resnet50':
            self.resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            expand = 4
        elif backbone == 'encoder:resnet50':
            self.resnet = models.resnet50(pretrained = True)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()


        self.upsamplex2 = nn.Upsample(scale_factor=2)
        
        self.learnable = learnable 
        
        self.upsamplex2l1_single = nn.ConvTranspose2d(256*expand, 256*expand, 4, 2, 1)
        
        if self.learnable:
            self.upsamplex2l1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
            self.upsamplex2l2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        # For ensuring the reproducility
        # refer to https://github.com/pytorch/pytorch/issues/7068#issuecomment-716719820
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        
        self.regressor = TwoLayerConv2d(in_channels=32, out_channels=1)
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
        if sigmoid3d==True:
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

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
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


class MTBIT(ResNet):
    def __init__(self, input_nc=3, output_nc=3, resnet_stages_num=4,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=8,
                 dim_head=64, decoder_dim_head=16,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable=False,
                 decoder_softmax=True,
                 ):

        super(MTBIT, self).__init__(input_nc, output_nc, backbone=backbone,
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

    def forward(self, x1, x2):
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
        

        return x2d, x3d

    
class MTBIT_levir(ResNet):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable=False,
                 decoder_softmax=True,
                 ):

        super(MTBIT_levir, self).__init__(input_nc, output_nc, backbone=backbone,
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
        decoder_pos_size = 512 // 4
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

    def forward(self, x1, x2):
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


        return x2d

class MTBIT_par_HE(ResNet):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, sigmoid3d=False
                 ):

        super(MTBIT_par_HE, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               learnable = learnable,sigmoid3d=sigmoid3d, par_HE=True)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 512//4
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
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
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
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
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2, output_height=True):
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
            x1 = self.upsamplex4(x1)

        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)
        x3d = self.active3d(x3d)

        if output_height:
            x3d_HE = self.regressor_HE(x1)
            x3d_HE = self.active3d(x3d_HE)
            return x2d, x3d, x3d_HE
        else:
            return x2d, x3d


class HE_module(ResNet):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=8, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable=False,
                 decoder_softmax=True, sigmoid3d=False
                 ):

        super(HE_module, self).__init__(input_nc, output_nc, backbone=backbone,
                                    resnet_stages_num=resnet_stages_num,
                                    if_upsample_2x=if_upsample_2x,
                                    learnable=learnable, sigmoid3d=sigmoid3d)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)
        self.learnable = learnable
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2 * dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len, 32))
        decoder_pos_size = 512 // 4
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

    def forward(self, x):
        # forward backbone resnet
        x = self.forward_single(x)

        #  forward tokenzier
        token = self._forward_semantic_tokens(x)
        # forward transformer encoder
        if self.token_trans:
            token = self._forward_transformer(token)

        # forward transformer decoder
        x = self._forward_transformer_decoder(x, token)

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

        return x2d, x3d

class MTBIT_serial_HE(ResNet):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable=False,
                 decoder_softmax=True, sigmoid3d=False
                 ):

        super(MTBIT_serial_HE, self).__init__(input_nc, output_nc, backbone=backbone,
                                    resnet_stages_num=resnet_stages_num,
                                    if_upsample_2x=if_upsample_2x,
                                    learnable=learnable, sigmoid3d=sigmoid3d)

        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)
        self.learnable = learnable
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2 * dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))
        decoder_pos_size = 512 // 4
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
        self.HEM = HE_module(input_nc=3, output_nc=2, token_len=8, resnet_stages_num=4, if_upsample_2x=True,
                enc_depth=1, dec_depth=8, decoder_dim_head=16)

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

    def forward(self, x1, x2):
        _, x1_height = self.HEM(x1)

        x1 = torch.concat([x1_height,x1_height,x1_height],dim=1)
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

        return x2d, x3d, x1_height