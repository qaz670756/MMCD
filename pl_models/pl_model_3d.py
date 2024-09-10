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

# class LitProgressBar(ProgressBar):


# define the LightningModule
class pl_trainer(pl_trainer_base):
    def __init__(self, model=None, exp_config=None, criterion2d=None, criterion3d=None, save_img=False, aux=False):
        super().__init__()

        self.model = model

        # self.model = MTBIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
        #            enc_depth=1, dec_depth=8,
        #            decoder_dim_head=16, backbone='resnet50')

        self.exp_config = exp_config
        self.optim_params = exp_config['optim']
        self.min_scale = exp_config['data']['min_value']
        self.max_scale = exp_config['data']['max_value']
        self.lweight2d, self.lweight3d = exp_config['model']['loss_weights']
        self.criterion2d = criterion2d
        self.criterion3d = criterion3d


        self.test_step_output = [0 for _ in range(6)]
        self.valid_step_output = [0 for _ in range(6)]
        self.bacth_idx = 0
        self.bacth_nochange = 0
        self.save_img = save_img
        self.aux = aux

        self.custom_lut = self.genColormap()

        self.record = []



    def cal_loss(self, out3d, mask3d, out2d=None, mask2d=None, aux=None):

        if type(out3d) == list:
            loss3d = 0.0
            for sub_out3d in out3d:
                loss3d += self.criterion3d(F.interpolate(sub_out3d, size=out3d[0].shape[2:]).squeeze(dim=1), mask3d)
        else:
            loss3d = self.criterion3d(out3d.squeeze(dim=1), mask3d)

        if out2d is not None:
            loss2d = self.criterion2d(out2d, mask2d.long())

        #import pdb;pdb.set_trace()
        if aux is not None:
            loss3d += self.criterion3d(F.interpolate(aux, size=out3d.shape[2:]).squeeze(dim=1), mask3d)

        loss = self.lweight2d * loss2d + self.lweight3d * loss3d

        return loss


    def training_step_func(self, batch):
        t1, t2, _, mask3d = batch
        mask3d = self.norm_train(mask3d)
        #import pdb;pdb.set_trace()
        if self.aux:
            out3d, out_aux = self.model(t1, t2, return_aux=True)
        else:
            out3d = self.model(t1, t2)
        return self.cal_loss(out3d, mask3d, aux=out_aux if self.aux else None)

    def training_step_func_plabel(self, batch):
        t1, t2, mask2d, mask3d = batch
        pmask2d = mask3d.clone().type(mask2d.dtype)
        pmask2d[pmask2d>0]=2
        pmask2d[pmask2d<0]=1

        out3d, outp2d = self.model(t1,t2)
        return self.cal_loss(out3d, mask3d,outp2d, pmask2d)



    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        if 'plabel' in self.exp_config['model']['model'].lower():
            loss =  self.training_step_func_plabel(batch)
        
        else:
            loss = self.training_step_func(batch)

        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        ############
        # import pdb;pdb.set_trace()

        t1, t2, mask2d, mask3d, img_path = batch

        if self.aux:
            out2d, out3d, out_aux = self.model(t1, t2)
        else:
            out2d, out3d = self.model(t1, t2)

        if type(out3d) == list:
            out3d = [self.norm_infer(_) for _ in out3d]
        else:
            out3d = self.norm_infer(out3d)

        loss = self.cal_loss(out2d, mask2d, out3d, mask3d, out_aux=out_aux if self.aux else None)
        if type(out2d) == list:
            out2d, out3d = out2d[0], out3d[0]

        out2d = out2d.detach().argmax(dim=1)

        # metric evalutation
        self.valid_step_output = self.eval_metrics(out3d, mask3d, out2d, mask2d, loss.item(),
                                                   self.valid_step_output)
        self.bacth_idx = batch_idx

    def on_validation_end(self):
        N = self.bacth_idx + 1
        mean_mae, rmse1, rmse2, rel, zncc, TN, FP, FN, TP, loss, chamferDist = self.valid_step_output
        mean_mae = mean_mae / N
        mIoU = TP / (TP + FN + FP + 1e-10)
        mean_f1 = 2 * TP / (2 * TP + FP + FN + 1e-10)
        RMSE1 = np.sqrt(rmse1 / N)
        RMSE2 = np.sqrt(rmse2 / (N - self.bacth_nochange))
        cRel = rel / (N - self.bacth_nochange)
        cZNCC = zncc / (N - self.bacth_nochange)
        loss /= N
        cd = chamferDist / N

        print('|metrics|mIoU|F1-score|ChamferDist|RMSE|MAE|cRMSE|cRel|cZNCC|loss|')
        print('|--|--|--|--|--|--|--|--|--|--|')
        print(
            f'|{mIoU * 100:.3f}|{mean_f1 * 100:.3f}|{cd:.3f}|{RMSE1:.3f}|{mean_mae:.3f}|{RMSE2:.3f}|{cRel:.3f}|{cZNCC:.3f}|{loss:.5f}|')


    def on_test_epoch_end(self):
        # import pdb;pdb.set_trace()
        N = self.bacth_idx + 1
        mean_mae, rmse1, rmse2, rel, zncc, chamferDist = self.test_step_output
        mean_mae = mean_mae / N
        RMSE1 = np.sqrt(rmse1 / N)
        RMSE2 = np.sqrt(rmse2 / (N - self.bacth_nochange))
        cRel = rel / (N - self.bacth_nochange)
        cZNCC = zncc / (N - self.bacth_nochange)
        cd = chamferDist / N
        

        
        print('|metrics|mIoU|F1-score|ChamferDist|RMSE|MAE|cRMSE|cRel|cZNCC|loss|')
        print('|--|--|--|--|--|--|--|--|--|--|')
        if 'plabel' in self.exp_config['model']['model'].lower():
            mIoU, mean_f1 = self.IoUMetric.compute_metrics()
            print(f'|{mIoU * 100:.3f}|{mean_f1 * 100:.3f}|{cd:.3f}|{RMSE1:.3f}|{mean_mae:.3f}|{RMSE2:.3f}|{cRel:.3f}|{cZNCC:.3f}|')
        else:
            print(f'|{cd:.3f}|{RMSE1:.3f}|{mean_mae:.3f}|{RMSE2:.3f}|{cRel:.3f}|{cZNCC:.3f}|')
        
        try:
            self.bins
            drawFig(self.total_gt_hist,self.total_hist,self.bins,self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1])+'/hist.png',self.num_bins)
        except:
            pass
            

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        t1, t2, mask2d, mask3d, self.img_path = batch
        if 'plabel' in self.exp_config['model']['model'].lower():
    
            out3d, outp2d = self.model(t1,t2)
            
        else:
            t1, t2, mask2d, mask3d, self.img_path = batch
            out3d = self.model(t1, t2)
        
        if type(out3d) == list:
            out3d = [self.norm_infer(_) for _ in out3d]
            out3d = out3d[0]
        else:
            out3d = self.norm_infer(out3d)

        # metric evalutation
        if 'plabel' in self.exp_config['model']['model'].lower():
            pmask2d = mask3d.clone().type(mask2d.dtype)
            pmask2d[pmask2d>0]=2
            pmask2d[pmask2d<0]=1
            outp2d = outp2d.detach().argmax(dim=1)
            self.test_step_output = self.eval_metrics_plabel(out3d, mask3d, outp2d, pmask2d,
                                                  self.test_step_output)
        else:
            self.test_step_output = self.eval_metrics(out3d, mask3d, mask2d,
                                                  self.test_step_output)

        if self.save_img and (mask2d >= 1).sum() > 0:

            targets = [
                        #self.model.TDec_x2.diff_c1,
                        # self.model.TDec_x2.soft_thresh.sigmoid,
                        # self.model.TDec_x2.dense_1x,
                        # self.model.TDec_x2.soft_thresh.fc2,
                        # self.model.TDec_x2.change_plabel
                        
                      ]
            names = ['soft_thresh_sigmoid','dense_1x','soft_thresh_fc2','change_plabel']
            for tid, target in enumerate(targets):
                vis_feature = gradCAM_vis(self.model.requires_grad_(), [target], torch.cat([t1,t2],dim=0), mask2d)
                self.save_vis_img_func(vis_feature, self.img_path, f'/out_vis_{names[tid]}')
            # f3d_out_path = os.path.join(self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]),'f3d')
            # os.makedirs(f3d_out_path, exist_ok=True)
            # save_imgtensor_func(out3d[0][0],out_dir=os.path.join(f3d_out_path,self.img_path[0].split('/')[-1]),mask=mask2d)

            self.mask3d = mask3d.detach().cpu().numpy()
            # #self.save_img_func(mask3d.detach().cpu().numpy()[0], self.img_path, '/out3d_gt')

            self.save_img_func(out3d.detach().cpu().numpy()[0][0], self.img_path, '/out3d')
            # self.save_img_func_2d(self.applyColor(outp2d.cpu().numpy()[0]).transpose(2, 0, 1), self.img_path,
            #                    f'/out2d', dim=3)
            
            mask = mask2d.detach().cpu().numpy()[0]!=0
            gt_hist, self.bins = getHist(mask3d.detach().cpu().numpy()[0][mask],self.num_bins)
            pred_hist, self.bins = getHist(out3d.detach().cpu().numpy()[0][0][mask],self.num_bins)
            self.total_hist += pred_hist
            self.total_gt_hist += gt_hist 

        self.bacth_idx = batch_idx

    
    def eval_metrics_plabel(self, out3d, mask3d, out2d, mask2d, results):
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
    

    def eval_metrics(self, out3d, mask3d, mask2d, results):

        eval_out3d = out3d.detach().cpu().numpy().ravel()
        eval_mask3d = mask3d.cpu().numpy().ravel()
        eval_mask2d = mask2d.cpu().numpy().ravel()
        eval_mask2d[eval_mask2d!=0] = 1

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


    def on_test_epoch_start(self):
        self.IoUMetric = IoUMetric(num_classes=3)
        self.num_bins = 200
        self.total_hist = np.zeros(200)
        self.total_gt_hist = np.zeros(200)



