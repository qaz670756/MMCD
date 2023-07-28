import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import torch
import numpy as np
import rasterio as ro, logging

import cv2
from sklearn import metrics
import metrics as other_metrics
from optim import set_scheduler

log = logging.getLogger()
log.setLevel(logging.ERROR)

# class LitProgressBar(ProgressBar):


# define the LightningModule
class pl_trainer(pl.LightningModule):
    def __init__(self, model=None, exp_config=None, criterion2d=None, criterion3d=None):
        super().__init__()
        
        
        self.model = model
        
        self.exp_config = exp_config
        self.optim_params = exp_config['optim']
        self.min_scale = exp_config['data']['min_value']
        self.max_scale = exp_config['data']['max_value']

        self.criterion2d = criterion2d

        try:
            self.norm_log = exp_config['data']['norm_log']
        except:
            self.norm_log = False
        print(f'Norm_log:{self.norm_log}')
            
        self.test_step_output = [0 for _ in range(5)]
        self.valid_step_output = [0 for _ in range(5)]
        self.bacth_idx = 0

        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        t1, t2, mask2d = batch
        
        

        out2d = self.model(t1,t2)
        
    
        loss = self.criterion2d(out2d, mask2d.long())
        
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'],
                prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        t1, t2, mask2d, mask3d = batch
        out2d, out3d = self.model(t1, t2)
        #import pdb;pdb.set_trace()
        out3d = self.norm_infer(out3d, self.norm_log)  # Tanh
        
        #mask3d = 2 * (mask3d - self.min_scale) / (self.max_scale - self.min_scale) - 1
    
        loss2d = self.criterion2d(out2d, mask2d.long())
        loss3d = self.criterion3d(out3d.squeeze(dim=1), mask3d)
        
        loss = self.lweight2d * loss2d + self.lweight3d * loss3d
        
        out2d = out2d.detach().argmax(dim=1)
        out2d = out2d.cpu().numpy()
        out3d = out3d.detach().cpu().numpy()

        # metric evalutation
        self.valid_step_output = self.eval_metrics(out3d, mask3d, out2d, mask2d, loss.item(),
                                                  self.valid_step_output)
        self.bacth_idx = batch_idx

    def on_validation_end(self):
        N = self.bacth_idx + 1
        mean_mae, rmse1, rmse2, TN, FP, FN, TP, loss = self.valid_step_output
        mean_mae = mean_mae / N
        mIoU = TP / (TP + FN + FP + 1e-10)
        mean_f1 = 2 * TP / (2 * TP + FP + FN + 1e-10)
        RMSE1 = np.sqrt(rmse1 / N)
        RMSE2 = np.sqrt(rmse2 / N)
        loss = loss/N
        
        self.logger.log_metrics({"mean_mae": mean_mae, "mIoU": mIoU,
                 "mean_f1": mean_f1, "RMSE1": RMSE1, "RMSE2": RMSE2})

        print(
            f'Validation metrics - 2D: F1 Score -> {mean_f1 * 100} %; mIoU -> {mIoU * 100} %; 3D: ' +
            f'MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m, loss:{loss}')

    def on_test_epoch_end(self):
        #import pdb;pdb.set_trace()
        N = self.bacth_idx + 1
        TN, FP, FN, TP, loss = self.test_step_output

        mIoU = TP / (TP + FN + FP + 1e-10)
        mean_f1 = 2 * TP / (2 * TP + FP + FN + 1e-10)
        loss /= N

        print(f'Testing metrics - 2D: F1 Score -> {mean_f1 * 100} %; mIoU -> {mIoU * 100} %; loss:{loss}')

    def test_step(self, batch, batch_idx):
        t1, t2, mask2d, img_path = batch
        out2d = self.model(t1, t2)

        
        loss = self.criterion2d(out2d, mask2d.long())
        
        out2d = out2d.detach().argmax(dim=1)
        out2d = out2d.cpu().numpy()

        # metric evalutation
        self.test_step_output = self.eval_metrics(out2d, mask2d, loss.item(),
                                                  self.test_step_output)
        
        
        self.save_img(np.uint8(out2d[0]*255), img_path, '/out2d')
        self.bacth_idx = batch_idx


    def eval_metrics(self, out2d, mask2d, loss, results):

        eval_mask2d = mask2d.cpu().numpy().ravel()

        try:
            tn, fp, fn, tp = metrics.confusion_matrix(eval_mask2d.ravel(), out2d.ravel()).ravel()
        except:
            tn, fp, fn, tp = [0, 0, 0, 0]


        eval_results = [tn,fp,fn,tp,loss]
        results = [x+y for x,y in zip(eval_results,results)]

        return results

    def save_img(self, img, img_path, save_dir=None, size=(512,512)):

        save_dir = self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]) + save_dir
        os.makedirs(save_dir, exist_ok=True)
        #img = cv2.resize(img,size)
        img_name = img_path[0].split('/')[-1]

        cv2.imwrite(os.path.join(save_dir, img_name),img)



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


