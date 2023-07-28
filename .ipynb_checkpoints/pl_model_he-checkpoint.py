import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

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

        self.test_step_output = [0 for _ in range(10)]
        self.valid_step_output = [0 for _ in range(10)]
        self.bacth_idx = 0

    # def forward(self, t1,t2) :
    #     return self.model(t1,t2)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        t1, t2, mask2d, mask3d, height = batch
        
        

        out2d, out3d, outheight = self.model(t1,t2)
        
        mask3d = 2 * (mask3d - self.min_scale) / (self.max_scale - self.min_scale) - 1
        height = 2 * (height - self.min_scale) / (self.max_scale - self.min_scale) - 1
    
        loss2d = self.criterion2d(out2d, mask2d.long())
        loss3d = self.criterion3d(out3d.squeeze(dim=1), mask3d)
        loss3d += self.criterion3d(outheight.squeeze(dim=1), height)
        
        loss = self.lweight2d * loss2d + self.lweight3d * loss3d
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("2d_loss", loss2d, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("3d_loss", loss3d, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'],
                prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        t1, t2, mask2d, mask3d, height = batch
        out2d, out3d, outheight = self.model(t1, t2)
        #import pdb;pdb.set_trace()
        out3d = (out3d + 1) * (self.max_scale - self.min_scale) / 2 + self.min_scale  # Tanh
        outheight = (outheight + 1) * (self.max_scale - self.min_scale) / 2 + self.min_scale
        #mask3d = 2 * (mask3d - self.min_scale) / (self.max_scale - self.min_scale) - 1
    
        loss2d = self.criterion2d(out2d, mask2d.long())
        loss3d = self.criterion3d(out3d.squeeze(dim=1), mask3d)
        loss3d += self.criterion3d(outheight.squeeze(dim=1), height)
        
        loss = self.lweight2d * loss2d + self.lweight3d * loss3d
        
        out2d = out2d.detach().argmax(dim=1)
        out2d = out2d.cpu().numpy()
        out3d = out3d.detach().cpu().numpy()
        outheight = outheight.detach().cpu().numpy()
        
        # metric evalutation
        self.valid_step_output = self.eval_metrics(out3d, mask3d, outheight, height, out2d, mask2d, loss.item(),
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
        mean_mae, rmse1, rmse2, mean_mae_height, rmse1_height, TN, FP, FN, TP, loss = self.test_step_output
        mean_mae = mean_mae / N
        mean_mae_height = mean_mae_height / N
        mIoU = TP / (TP + FN + FP + 1e-10)
        mean_f1 = 2 * TP / (2 * TP + FP + FN + 1e-10)
        RMSE1 = np.sqrt(rmse1 / N)
        RMSE2 = np.sqrt(rmse2 / N)
        RMSE1_height = np.sqrt(rmse1_height / N)
        loss /= N

        print(
            f'Testing metrics - 2D: F1 Score -> {mean_f1 * 100} %; mIoU -> {mIoU * 100} %; 3D: ' +
            f'MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m,  loss:{loss}'+
        f'MAE_height -> {mean_mae_height} m; RMSE_height -> {RMSE1_height} m')

    def test_step(self, batch, batch_idx):
        t1, t2, mask2d, mask3d, height, img_path = batch
        out2d, out3d, outheight = self.model(t1, t2)
        #import pdb;pdb.set_trace()
        out3d = (out3d + 1) * (self.max_scale - self.min_scale) / 2 + self.min_scale  # Tanh
        outheight = (outheight + 1) * (self.max_scale - self.min_scale) / 2 + self.min_scale
        #mask3d = 2 * (mask3d - self.min_scale) / (self.max_scale - self.min_scale) - 1
    
        loss2d = self.criterion2d(out2d, mask2d.long())
        loss3d = self.criterion3d(out3d.squeeze(dim=1), mask3d)
        loss3d += self.criterion3d(outheight.squeeze(dim=1), height)
        
        loss = self.lweight2d * loss2d + self.lweight3d * loss3d
        
        out2d = out2d.detach().argmax(dim=1)
        out2d = out2d.cpu().numpy()
        out3d = out3d.detach().cpu().numpy()
        outheight = outheight.detach().cpu().numpy()

        # metric evalutation
        self.test_step_output = self.eval_metrics(out3d, mask3d, outheight, height, out2d, mask2d, loss.item(),
                                                  self.test_step_output)
        
        
        #self.save_img(np.uint8(out2d[0]*255), img_path, '/out2d')
        #self.save_img(outheight[0][0], img_path, '/height')
        #self.save_img(out3d[0][0], img_path, '/out3d')
        self.bacth_idx = batch_idx


    def eval_metrics(self, out3d, mask3d, outheight, height, out2d, mask2d, loss, results):
        eval_out3d = out3d.ravel()
        eval_outheight = outheight.ravel()
        
        eval_mask3d = mask3d.cpu().numpy().ravel()
        eval_height = height.cpu().numpy().ravel()
        eval_mask2d = mask2d.cpu().numpy().ravel()

        try:
            tn, fp, fn, tp = metrics.confusion_matrix(eval_mask2d.ravel(), out2d.ravel()).ravel()
        except:
            tn, fp, fn, tp = [0, 0, 0, 0]

        mean_ae = metrics.mean_absolute_error(eval_mask3d, eval_out3d)
        mean_ae_height = metrics.mean_absolute_error(eval_height, eval_outheight)

        s_rmse1 = other_metrics.metric_mse(eval_out3d, eval_mask3d, eval_mask2d, exclude_zeros=False)
        s_rmse2 = other_metrics.metric_mse(eval_out3d, eval_mask3d, eval_mask2d, exclude_zeros=True)

        s_rmse1_height = other_metrics.metric_mse(eval_outheight, eval_height, eval_mask2d, exclude_zeros=False)
        
        #max_error = metrics.max_error(mask3d.ravel(), out3d.ravel())
        #mask_max = np.abs(mask3d.cpu().numpy()).max()

        eval_results = [mean_ae,s_rmse1,s_rmse2,mean_ae_height,s_rmse1_height,tn,fp,fn,tp,loss]
        results = [x+y for x,y in zip(eval_results,results)]

        return results

    def save_img(self, img, img_path, save_dir=None, size=(512,512)):
        #if not save_dir:
        save_dir = self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]) + save_dir
        os.makedirs(save_dir, exist_ok=True)
        #img = cv2.resize(img,size)
        img_name = img_path[0].split('/')[-1]
        src = ro.open(img_path[0])
        with ro.open(os.path.join(save_dir, img_name), mode='w', driver='GTiff', 
                     width=size[0], height=size[1],
                     count=1, crs=src.crs, transform=src.transform, dtype=img.dtype) as dst:  #
            dst.write(img, indexes=1)



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


