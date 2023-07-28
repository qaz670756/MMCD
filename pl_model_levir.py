import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import torch
import numpy as np
import rasterio as ro, logging
import torch.nn.functional as F
import cv2
from sklearn import metrics
import metrics as other_metrics
from optim import set_scheduler

log = logging.getLogger()
log.setLevel(logging.ERROR)
from utils.mIoU import IoUMetric

# class LitProgressBar(ProgressBar):


# define the LightningModule
class pl_trainer(pl.LightningModule):
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
            
        self.test_step_output = [0 for _ in range(5)]
        self.valid_step_output = [0 for _ in range(5)]
        self.bacth_idx = 0
        self.bacth_nochange = 0
        self.save_img = save_img
        self.aux = aux
        #import pdb;pdb.set_trace()
        self.chamferDist = other_metrics.metric_chamferDist()


        
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
        
        loss = self.lweight2d * loss2d
        
        return loss
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        t1, t2, mask2d = batch
        
        

        out2d = self.model(t1,t2)
        #
        if type(out2d)!=torch.tensor:
            loss = 0
            #import pdb;pdb.set_trace()
            for sub_out2d in out2d:
                loss += self.cal_loss(sub_out2d, mask2d.long())
            loss *= 0.5
            
        else:
            loss = self.cal_loss(out2d, mask2d.long())
        
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'],
                prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_test_epoch_start(self):
        self.IoUMetric = IoUMetric(num_classes=2)

    def on_test_epoch_end(self):
        #import pdb;pdb.set_trace()
        N = self.bacth_idx + 1

        mIoU, mean_f1 = self.IoUMetric.compute_metrics()


        print('|mIoU|F1-score|')
        print('|--|--|')
        print(f'|{mIoU * 100:.3f}|{mean_f1 * 100:.3f}|')

    def test_step(self, batch, batch_idx):
        t1, t2, mask2d, self.img_path = batch
        if self.aux:
            out2d, out_aux = self.model(t1,t2)
        else:
            out2d = self.model(t1,t2)

        #mask3d = 2 * (mask3d - self.min_scale) / (self.max_scale - self.min_scale) - 1
        
        #loss = self.cal_loss(out2d, mask2d, out_aux=out_aux if self.aux else None)
        if type(out2d)!=torch.tensor:
            out2d = out2d[0]
        
        
        out2d = out2d.detach().argmax(dim=1)

        #out2d = out2d
        #out3d = out3d

        # metric evalutation
        self.test_step_output = self.eval_metrics(out2d, mask2d, #loss.item(),
                                                  self.test_step_output)
        
        if self.save_img and (mask2d>=1).sum()>0:
            # np.uint8(out2d.cpu().numpy()[0]*255)
            self.save_img_func(self.applyColor(out2d.cpu().numpy()[0]), self.img_path, '/out2d',dim=3)
            #self.save_img_func(out3d.detach().cpu().numpy()[0][0], self.img_path, '/out3d')
        self.bacth_idx = batch_idx


    def applyColor(self, pred):

        pred = np.stack([pred, pred, pred], 2)
        pred = np.uint8(pred)

        color_list = [[130,217,178],[191,158,142], [250,137,137]]

        # generate color for pred and gt
        for cls_id, color in enumerate(color_list):
            for cid, cvalue in enumerate(color):
                pred[:, :, cid][pred[:, :, cid] == cls_id + 1] = cvalue
        
        return pred

    def eval_metrics(self, out2d, mask2d, results):
        eval_out2d = out2d.cpu().numpy()
        eval_mask2d = mask2d.cpu().numpy().ravel()

        self.IoUMetric.process(eval_mask2d.ravel(), eval_out2d.ravel())

    def save_img_func(self, img, img_path, save_dir=None, size=(1024,1024),dim=1):
        #if not save_dir:
        save_dir = self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]) + save_dir
        os.makedirs(save_dir, exist_ok=True)
        #img = cv2.resize(img,size)
        img_name = img_path[0].split('/')[-1]
        src = ro.open(img_path[0])
        with ro.open(os.path.join(save_dir, img_name), mode='w', driver='GTiff', 
                     width=size[0], height=size[1],
                     count=dim, crs=src.crs, transform=src.transform, dtype=img.dtype) as dst:  #
            #import pdb;pdb.set_trace()
            dst.write(img.transpose(2,0,1))#, indexes=3)



    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                lr = self.optim_params['lr'], 
                                betas = self.optim_params['beta'],
                                weight_decay= self.optim_params['weight_decay'])
        scheduler = {
            "scheduler":set_scheduler(self.exp_config['optim'], optimizer),
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [scheduler]


