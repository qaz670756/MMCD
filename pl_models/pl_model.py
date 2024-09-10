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
from utils.vis_tsne import visualize_tsne, visualize_tsne_multimodal

log = logging.getLogger()
log.setLevel(logging.ERROR)



# define the LightningModule
class pl_trainer(pl_trainer_base):
    def __init__(self, model=None, exp_config=None, criterion2d=None,
                 criterion3d=None, save_img=False, aux=False):
        super().__init__()

        self.model = model

        self.exp_config = exp_config
        self.optim_params = exp_config['optim']
        self.min_scale = exp_config['data']['min_value']
        self.max_scale = exp_config['data']['max_value']
        self.lweight2d, self.lweight3d = exp_config['model']['loss_weights']
        self.criterion2d = criterion2d
        self.criterion3d = criterion3d

       

        self.test_step_output = [0 for _ in range(6)]
        self.valid_step_output = [0 for _ in range(6)]
        self.batch_idx = 0
        self.bacth_nochange = 0
        self.save_img = save_img
        self.aux = aux
        

        self.custom_lut = self.genColormap()
        self.sigma = 1



    def cal_loss(self, out2d, mask2d, out3d, mask3d, out_aux=None):

        if type(out2d) == list:
            loss2d = 0.0
            for sub_out2d in out2d:
                loss2d += self.criterion2d(F.interpolate(sub_out2d, size=out2d[0].shape[2:]), mask2d.long())
        else:
            loss2d = self.criterion2d(out2d, mask2d.long())
        if self.aux:
            # 0.4 is the weight of auxiliary loss from P2VNet:https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9975266
            loss2d += 0.4 * self.criterion2d(out_aux, mask2d.long())
        if type(out3d) == list:
            loss3d = 0.0
            for sub_out3d in out3d:
                loss3d += self.criterion3d(F.interpolate(sub_out3d, size=out3d[0].shape[2:]).squeeze(dim=1), mask3d)
        else:
            # import pdb;pdb.set_trace()
            loss3d = self.criterion3d(out3d.squeeze(dim=1), mask3d)
        if 'dynamicweight' in self.exp_config['model']:
            lweight = 1 - (self.current_epoch + 1) / self.exp_config['optim']['num_epochs']
            loss = lweight * loss2d + self.sigma * (1 - lweight) * loss3d
        else:
            loss = self.lweight2d * loss2d + self.lweight3d * loss3d

        return loss, loss2d.item(), loss3d.item()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        t1, t2, mask2d, mask3d = batch
        mask3d = self.norm_train(mask3d)

        if self.aux:
            out2d, out3d, out_aux = self.model(t1, t2)
        else:
            out2d, out3d = self.model(t1, t2)

        loss, loss2d, loss3d = self.cal_loss(out2d, mask2d, out3d, mask3d, out_aux=out_aux if self.aux else None)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("2d_loss", loss2d, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("3d_loss", loss3d, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_test_epoch_end(self):

        N = self.batch_idx + 1
        mean_mae, rmse1, rmse2, rel, zncc, chamferDist = self.test_step_output
        mean_mae = mean_mae / N
        RMSE1 = np.sqrt(rmse1 / N)
        RMSE2 = np.sqrt(rmse2 / (N - self.bacth_nochange))
        cRel = rel / (N - self.bacth_nochange)
        cZNCC = zncc / (N - self.bacth_nochange)
        cd = chamferDist / N
        mIoU, mean_f1 = self.IoUMetric.compute_metrics()

        print('|metrics|mIoU|F1-score|ChamferDist|RMSE|MAE|cRMSE|cRel|cZNCC|')
        print('|--|--|--|--|--|--|--|--|--|--|')
        print(
            f'|{mIoU * 100:.3f}|{mean_f1 * 100:.3f}|{cd:.3f}|{RMSE1:.3f}|{mean_mae:.3f}|{RMSE2:.3f}|{cRel:.3f}|{cZNCC:.3f}|')
        try:
            self.bins
            drawFig(self.total_gt_hist,self.total_hist,self.bins,
                    self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1])+'/hist.svg',self.num_bins)
        except:
            pass

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        t1, t2, mask2d, mask3d, self.img_path = batch
        if self.aux:
            out2d, out3d, out_aux = self.model(t1, t2)
        else:
            out2d, out3d = self.model(t1, t2)

        out3d_copy = out3d.clone()
        if type(out3d) == list:
            out3d = [self.norm_infer(_) for _ in out3d]
        else:
            out3d = self.norm_infer(out3d)

        if type(out2d) == list:
            out2d, out3d = out2d[0], out3d[0]
            

        out2d = out2d.detach().argmax(dim=1)

        # metric evalutation
        self.test_step_output = self.eval_metrics(out3d, mask3d, out2d, mask2d,
                                                  self.test_step_output)

        if self.save_img and (mask2d >= 1).sum() > 0:

            targets = [
                        #self.model.TDec_x2.diff_c1,
                        #self.model.TDec_x2.dense_1x
                        #self.model.TDec_x2.dense_1x
                        
                      ]
            names = ['dense_1x']
            for tid, target in enumerate(targets):
                vis_feature = gradCAM_vis(self.model.requires_grad_(), [target], torch.cat([t1,t2],dim=0), mask2d)
                self.save_vis_img_func(vis_feature, self.img_path, f'/out_vis_{names[tid]}')
            #import pdb;pdb.set_trace()

            tsne_path = self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]) + 'tsne_multimodal/'
            os.makedirs(tsne_path, exist_ok=True)
            # if 'rotterdam_18_02' in self.img_path[0] or 'amsterdam_18_13' in self.img_path[0] or \
            #         'rotterdam_22_13' in self.img_path[0] or 'rotterdam_22_15' in self.img_path[0]:
            #     # visualize_tsne(vis_feature, mask2d,
            #     #            tsne_path + self.img_path[0].split('/')[-1].replace('.tif','.png'), total=100000)
            #     visualize_tsne_multimodal(vis_feature,
            #                               tsne_path + self.img_path[0].split('/')[-1].replace('.tif', '.png'),
            #                               total=100000)

            mask = mask2d.detach().cpu().numpy()[0]!=0
            gt_hist, self.bins = getHist(mask3d.detach().cpu().numpy()[0][mask],self.num_bins)
            pred_hist, self.bins = getHist(out3d.detach().cpu().numpy()[0][0][mask],self.num_bins)
            self.total_hist += pred_hist
            self.total_gt_hist += gt_hist 

            f3d_out_path = os.path.join(self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]),'f3d')
            os.makedirs(f3d_out_path,exist_ok=True)
            save_imgtensor_func(out3d[0][0],out_dir=os.path.join(f3d_out_path,self.img_path[0].split('/')[-1]),mask=mask2d)

            self.mask3d = mask3d.detach().cpu().numpy()
            self.save_img_func(self.applyColor(out2d.cpu().numpy()[0]).transpose(2, 0, 1), self.img_path, '/out2d',
                               dim=3)
            self.save_img_func_3d(out3d.detach().cpu().numpy()[0][0], self.img_path, '/out3d')
            self.save_img_func_3d(mask3d.detach().cpu().numpy()[0], self.img_path, '/gt3d')
            #self.save_img_func_3d(out3d.sigmoid().detach().cpu().numpy()[0][0], self.img_path, '/out3d_sigmoid')
            #self.save_img_func_3d(out3d_copy.sigmoid().detach().cpu().numpy()[0][0], self.img_path, '/out3d_noNorm_sigmoid')
        self.batch_idx = batch_idx

    def on_test_epoch_start(self):
        self.IoUMetric = IoUMetric(num_classes=3)
        self.num_bins = 200
        self.total_hist = np.zeros(200)
        self.total_gt_hist = np.zeros(200)

    def eval_metrics(self, out3d, mask3d, out2d, mask2d, results):
        eval_out2d = out2d.cpu().numpy()
        eval_out3d = out3d.detach().cpu().numpy().ravel()
        eval_mask3d = mask3d.cpu().numpy().ravel()
        eval_mask2d = mask2d.cpu().numpy().ravel()
        eval_mask2d[eval_mask2d == 3] = 2
        eval_out2d[eval_out2d == 3] = 2
        self.IoUMetric.process(eval_mask2d.ravel(), eval_out2d.ravel())
        eval_mask2d[eval_mask2d > 0] = 1
        eval_out2d[eval_out2d > 0] = 1

        mean_ae = metrics.mean_absolute_error(eval_mask3d, eval_out3d)

        s_rmse1 = other_metrics.metric_mse(eval_out3d, eval_mask3d, eval_mask2d, exclude_zeros=False)
        s_rmse2 = other_metrics.metric_mse(eval_out3d, eval_mask3d, eval_mask2d, exclude_zeros=True)

        rel = other_metrics.metric_rel(eval_out3d, eval_mask3d, eval_mask2d)
        zncc = other_metrics.metric_ncc(eval_out3d, eval_mask3d, eval_mask2d)

        chamferDist = 0  # self.chamferDist.func(out3d, mask3d)
        if zncc == 0:
            self.bacth_nochange += 1

        eval_results = [mean_ae, s_rmse1, s_rmse2, rel, zncc, chamferDist]
        results = [x + y for x, y in zip(eval_results, results)]

        return results







    

