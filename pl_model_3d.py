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
from utils.mIoU import IoUMetric

log = logging.getLogger()
log.setLevel(logging.ERROR)
import cv2


# class LitProgressBar(ProgressBar):


# define the LightningModule
class pl_trainer(pl.LightningModule):
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
        try:
            self.norm_log = exp_config['data']['norm_log']
        except:
            self.norm_log = False
        print(f'Norm_log:{self.norm_log}')

        self.test_step_output = [0 for _ in range(6)]
        self.valid_step_output = [0 for _ in range(6)]
        self.bacth_idx = 0
        self.bacth_nochange = 0
        self.save_img = save_img
        self.aux = aux
        # import pdb;pdb.set_trace()
        self.chamferDist = other_metrics.metric_chamferDist()

        self.custom_lut = self.genColormap()
    # def forward(self, t1,t2) :
    #     return self.model(t1,t2)
        self.record = []
    def norm(self, x, M, m):
        return (x - m) / (M - m)

    def reverse_norm(self, x, M, m):
        return x * (M - m) + m

    def norm_train(self, x, log=False):
        if type(x) == np.ndarray:
            log_func = np.log
        else:
            log_func = torch.log

        if log:
            mm = 27.29
            MM = 83.26
            M = np.log(MM)
            m = -np.log(mm)

            idx_pos = x > 0
            idx_neg = x < 0

            # print(f'before norm x_shape:{x[idx_pos].shape,x[idx_neg].shape}')

            x_pos = self.norm(log_func(x[idx_pos]), M, m)
            x_neg = -self.norm(log_func(-x[idx_neg]), M, m)

            # print(f'after norm x_shape:{x_pos.shape,x_neg.shape}')

            x[idx_pos] = x_pos
            x[idx_neg] = x_neg

            return x

        else:

            return 2 * (x - self.min_scale) / (self.max_scale - self.min_scale) - 1

    def norm_infer(self, x, log=False):
        if type(x) == np.ndarray:
            Exp = np.exp
        else:
            Exp = torch.exp

        if log:
            mm = 27.29
            MM = 83.26
            M = np.log(MM)
            m = -np.log(mm)

            idx_pos = x > 0
            idx_neg = x < 0

            x_pos = Exp(self.reverse_norm(x[idx_pos], M, m))
            x_neg = -Exp(self.reverse_norm(-x[idx_neg], M, m))

            x[idx_pos] = x_pos
            x[idx_neg] = x_neg

            return x
        else:
            return (x + 1) * (self.max_scale - self.min_scale) / 2 + self.min_scale

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
        mask3d = self.norm_train(mask3d, self.norm_log)
        #import pdb;pdb.set_trace()
        if self.aux:
            out3d, out_aux = self.model(t1, t2, return_aux=True)
        else:
            out3d = self.model(t1, t2)
        return self.cal_loss(out3d, mask3d, aux=out_aux if self.aux else None)

    def training_step_func_plabel(self, batch):
        t1, t2, _, mask3d = batch
        pmask2d = mask3d.clone().type(mask2d.dtype)
        pmask2d[pmask2d>0]=2
        pmask2d[pmask2d<0]=1

        out3d, outp2d = self.model(t1,t2)
        return self.cal_loss(out3d, mask3d,outp2d, pmask2d)



    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        if 'plabel' in self.exp_config['model']['model'].lower():
            loss =  self.training_step_plabel(batch)
        
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
            out3d = [self.norm_infer(_, self.norm_log) for _ in out3d]
        else:
            out3d = self.norm_infer(out3d, self.norm_log)

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
        if self.save_img:
            rc1 = [x for x in self.record[0]]
            rc2 = [x for x in self.record[1]]
            
            print('rc1:',np.mean(rc1),np.max(rc1),np.min(rc1))
            print('rc2:',np.mean(rc2),np.max(rc2),np.min(rc2))


        print('|metrics|mIoU|F1-score|ChamferDist|RMSE|MAE|cRMSE|cRel|cZNCC|loss|')
        print('|--|--|--|--|--|--|--|--|--|--|')
        print(f'|{cd:.3f}|{RMSE1:.3f}|{mean_mae:.3f}|{RMSE2:.3f}|{cRel:.3f}|{cZNCC:.3f}|')

    def test_step(self, batch, batch_idx):
        t1, t2, mask2d, mask3d, self.img_path = batch

        out3d = self.model(t1, t2)

        # mask3d = 2 * (mask3d - self.min_scale) / (self.max_scale - self.min_scale) - 1
        if type(out3d) == list:
            out3d = [self.norm_infer(_, self.norm_log) for _ in out3d]
        else:
            out3d = self.norm_infer(out3d, self.norm_log)

        if type(out3d) == list:
            out3d = out3d[0]


        # out2d = out2d
        # out3d = out3d

        # metric evalutation
        self.test_step_output = self.eval_metrics(out3d, mask3d, mask2d,
                                                  self.test_step_output)

        if self.save_img and (mask2d >= 1).sum() > 0:
            #import pdb;pdb.set_trace()
            self.mask3d = mask3d.detach().cpu().numpy()
            self.save_img_func(mask3d.detach().cpu().numpy()[0], self.img_path, '/out3d_gt')

            self.save_img_func(out3d.detach().cpu().numpy()[0][0], self.img_path, '/out3d')
            
        self.bacth_idx = batch_idx

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

    def genColormap(self):
        custom_lut = []  # np.zeros((256, 1, 3), dtype=np.uint8)

        def convert_line(line):
            return [int(x) for x in line.split(',')[1:4]]

        lines = open('./utils/colormap.txt', 'r').readlines()
        for idx in range(len(lines) - 1):
            lcolor = convert_line(lines[idx])
            rcolor = convert_line(lines[idx + 1])
            if idx == 0:
                custom_lut.append(lcolor)

            R = np.linspace(lcolor[0], rcolor[0], 6, dtype=int)[1:]
            G = np.linspace(lcolor[1], rcolor[1], 6, dtype=int)[1:]
            B = np.linspace(lcolor[2], rcolor[2], 6, dtype=int)[1:]

            for r, g, b in zip(R, G, B):
                custom_lut.append([r, g, b])

        # import pdb;pdb.set_trace()
        return np.array(custom_lut, dtype=np.uint8).reshape(256, 1, 3)
    def vis_3d(self, img, custom_lut, mM=None):
        if mM is None:
            m = img.min()#27.29
            M = img.max()#83.26 
        else:
            m, M = mM     
        self.record.append((m,M))
        # print(img.max(),img.min(),img.shape)
        img_gray = np.uint8(255 * self.norm(img, M, m))
        img_gray = np.stack([img_gray, img_gray, img_gray], axis=2)

        # print(img_gray.max(),img_gray.min())

        img_color = cv2.LUT(img_gray, custom_lut)  # cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
        # return cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
        return img_color

    def save_img_func(self, img, img_path, save_dir=None, size=(1024, 1024)):
        # if not save_dir:
        save_dir = self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]) + save_dir
        os.makedirs(save_dir, exist_ok=True)
        # img = cv2.resize(img,size)
        img_name = img_path[0].split('/')[-1]
        src = ro.open(img_path[0])
        # with ro.open(os.path.join(save_dir, img_name), mode='w', driver='GTiff',
        #              width=size[0], height=size[1],
        #              count=3, crs=src.crs, transform=src.transform, dtype=img.dtype) as dst:  #
        #     dst.write(self.vis_3d(img,self.custom_lut).transpose(2,0,1))
        color_img = self.vis_3d(img,self.custom_lut,(self.mask3d.min(),self.mask3d.max()))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, img_name),color_img)

    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=self.optim_params['lr'],
                                betas=self.optim_params['beta'],
                                weight_decay=self.optim_params['weight_decay'])
        scheduler = {
            "scheduler": set_scheduler(self.exp_config['optim'], optimizer),
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [scheduler]


