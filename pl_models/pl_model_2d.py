import os
import cv2

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np
import rasterio as ro, logging
from sklearn import metrics

from utils.mIoU import IoUMetric
from utils.optim import set_scheduler
from utils.vis_feature import gradCAM_vis, save_imgtensor_func
from utils.evaluation import getHist,drawFig
import utils.metrics as other_metrics

from pl_models.common import pl_trainer_base

log = logging.getLogger()
log.setLevel(logging.ERROR)



# define the LightningModule
class pl_trainer(pl_trainer_base):
    def __init__(self, model=None, exp_config=None, criterion2d=None, criterion3d=None, save_img=False, aux=False):
        super().__init__()
        
        
        self.model = model
        
        #self.model = MTBIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True, 
        #            enc_depth=1, dec_depth=8, 
        #            decoder_dim_head=16, backbone='resnet50')
        
        self.exp_config = exp_config
        self.optim_params = exp_config['optim']
        self.min_scale = exp_config['data']['min_value']
        self.max_scale = exp_config['data']['max_value']
        self.lweight2d, self.lweight3d = exp_config['model']['loss_weights']
        self.criterion2d = criterion2d
        self.criterion3d = criterion3d
        try:
            self.norm_log = exp_config['data']['norm_log']
        except:
            self.norm_log = False
        print(f'Norm_log:{self.norm_log}')
            
        self.bacth_idx = 0
        self.bacth_nochange = 0
        self.save_img = save_img
        self.aux = aux
        #import pdb;pdb.set_trace()
        
        
    


    
    def cal_loss(self, out2d, mask2d, out_aux=None):
        
        #import pdb;pdb.set_trace()
        if type(out2d)==list:
            loss2d = 0.0
            for sub_out2d in out2d:
                loss2d += self.criterion2d(F.interpolate(sub_out2d,size=out2d[0].shape[2:]), mask2d.long())
        else:
            loss2d = self.criterion2d(out2d, mask2d.long())
        if self.aux:
            # 0.4 is the weight of auxiliary loss from P2VNet:https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9975266
            loss2d += 0.4 * self.criterion2d(out_aux, mask2d.long())

        
        loss = loss2d
        
        return loss
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        t1, t2, mask2d, _ = batch
        #import pdb;pdb.set_trace()
        
        if self.aux:
            out2d, out_aux = self.model(t1,t2)
        else:
            out2d = self.model(t1,t2)
            
        loss = self.cal_loss(out2d, mask2d, out_aux=out_aux if self.aux else None)
        

        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #self.log("2d_loss", loss2d, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #self.log("3d_loss", loss3d, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'],
                prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        ############
        t1, t2, mask2d, mask3d, self.img_path = batch
        if self.aux:
            out2d, out_aux = self.model(t1,t2)
        else:
            out2d= self.model(t1,t2)


        loss = self.cal_loss(out2d, mask2d, out_aux=out_aux if self.aux else None)
        if type(out2d)==list:
            out2d = out2d[0]
        
        out2d = out2d.detach().argmax(dim=1)


        # metric evalutation
        self.valid_step_output = self.eval_metrics(out2d, mask2d, loss.item(),
                                                  self.valid_step_output)
        
        self.bacth_idx = batch_idx
        
    def on_validation_end(self):
        N = self.bacth_idx + 1
        TN, FP, FN, TP, loss = self.valid_step_output
        mIoU = TP / (TP + FN + FP + 1e-10)
        mean_f1 = 2 * TP / (2 * TP + FP + FN + 1e-10)
        loss /= N


        print('|metrics|mIoU|F1-score|loss|')
        print('|--|--|--|--|--|--|--|--|--|--|')
        print(f'|{mIoU * 100:.4f}|{mean_f1 * 100:.4f}|{loss:.5f}|')
    def on_test_epoch_start(self):
        self.IoUMetric = IoUMetric(num_classes=3)
        
    def on_test_epoch_end(self):
        #import pdb;pdb.set_trace()
        N = self.bacth_idx + 1

        mIoU, mean_f1 = self.IoUMetric.compute_metrics()


        print('|mIoU|F1-score|')
        print('|--|--|')
        print(f'|{mIoU * 100:.4f}|{mean_f1 * 100:.4f}|')

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        t1, t2, mask2d, _, self.img_path = batch
        if self.aux:
            out2d, out_aux = self.model(t1,t2)
        else:
            out2d = self.model(t1,t2)
        
        
        if type(out2d)==list:
            out2d = out2d[0]
        
        out2d = out2d.detach().argmax(dim=1)

        # metric evalutation
        self.test_step_output = self.eval_metrics(out2d, mask2d)
        
        if self.save_img and (mask2d>=1).sum()>0:
            targets = [
                        #self.model.TDec_x2.diff_c1,
                        self.model.TDec_x2.dense_1x
                        
                      ]
            names = ['dense_1x']
            for tid, target in enumerate(targets):
                vis_feature = gradCAM_vis(self.model.requires_grad_(), [target], torch.cat([t1,t2],dim=0), mask2d)
                self.save_vis_img_func(vis_feature, self.img_path, f'/out_vis_{names[tid]}')

            self.save_img_func(self.applyColor(out2d.cpu().numpy()[0]), self.img_path, '/out2d',dim=3)

        self.bacth_idx = batch_idx
        

    def eval_metrics(self, out2d, mask2d):
        eval_out2d = out2d.cpu().numpy()
        eval_mask2d = mask2d.cpu().numpy().ravel()
        eval_mask2d[eval_mask2d==3] = 2
        eval_out2d[eval_out2d==3] = 2
        self.IoUMetric.process(eval_mask2d.ravel(), eval_out2d.ravel())
        

