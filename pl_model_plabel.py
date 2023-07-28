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
        #self.chamferDist = other_metrics.metric_chamferDist()

        self.custom_lut = self.genColormap()
    # def forward(self, t1,t2) :
    #     return self.model(t1,t2)

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

    def cal_loss(self, out2d=[], mask2d = [], outp2d=None, pmask2d=None, out3d=None, mask3d=None):

        
        loss2d = 0.0
        loss3d = 0.0
        loss_consistency = 0.0
        #import pdb;pdb.set_trace()
        weights_2d = [1-0.2*i for i in range(len(out2d))]
        for i,sub_out2d in enumerate(out2d):
            loss2d += weights_2d[i]*self.criterion2d(F.interpolate(sub_out2d, size=out2d[0].shape[2:]), mask2d.long())


        if pmask2d is not None:
            loss2d += self.criterion2d(outp2d[0], pmask2d.long())
            
            overlap = mask2d==pmask2d
            overlap = overlap.unsqueeze(1)
            loss_consistency = torch.mean((overlap * (out2d[0]-outp2d[0]))**2)
        
        if self.aux:
            # 0.4 is the weight of auxiliary loss from P2VNet:https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9975266
            loss2d += 0.4 * self.criterion2d(out_aux, mask2d.long())

        weights = [1-0.2*i for i in range(len(out3d))]
        for i,sub_out3d in enumerate(out3d):
            loss3d += weights[i]*self.criterion3d(F.interpolate(sub_out3d, size=out3d[0].shape[2:]).squeeze(dim=1), mask3d)


        CLweight = (self.current_epoch + 1) / self.exp_config['optim']['num_epochs']
        if self.current_epoch<self.exp_config['optim']['num_epochs']//2:
            CLweight *=2
        else:
            CLweight = 1
        loss = self.lweight2d * loss2d + self.lweight3d * loss3d + CLweight * loss_consistency

        return loss, loss2d.detach().item(), loss3d.detach().item(), loss_consistency.detach().item()

    def toTuple(self, x):
        if type(x)==torch.Tensor:
            return [x]
        else:
            return list(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        t1, t2, mask2d, mask3d, plabel= batch
        mask3d = self.norm_train(mask3d, self.norm_log)
        

        out3d, out2d, outp2d = self.model(t1, t2)
        #import pdb;pdb.set_trace()
        loss,loss2d,loss3d,loss_consistency = self.cal_loss(out2d=self.toTuple(out2d), out3d=self.toTuple(out3d), 
                                                            outp2d=self.toTuple(outp2d),
                                                            mask2d = mask2d, mask3d=mask3d,  pmask2d=plabel)
        #import pdb;pdb.set_trace()
            
        # Logging to TensorBoard (if installed) by default
        self.log("loss_consistency", loss_consistency, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
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

    def test_step(self, batch, batch_idx):
        
        t1, t2, mask2d, mask3d, plabel, self.img_path = batch
        
        if self.aux:
            out2d, out3d, out_aux = self.model(t1, t2)
        else:
            out3d, out2d, outp2d = self.model(t1, t2)

        out3d = self.toTuple(out3d)
        out2d = self.toTuple(out2d)


        out3d = [self.norm_infer(_, self.norm_log) for _ in out3d]
        out3d_show = [F.interpolate(sub_out3d, size=out3d[0].shape[2:]) for sub_out3d in out3d]
        out3d = out3d[0]

        out2d_show = [F.interpolate(x,size=out2d[0].shape[2:]).detach().argmax(dim=1) for x in out2d]
        out2d= out2d[0]
        out2d = out2d.detach().argmax(dim=1)

        # metric evalutation
        self.test_step_output = self.eval_metrics(out3d, mask3d, out2d, mask2d,
                                                  self.test_step_output)

        if self.save_img and (mask2d >= 1).sum() > 0:
            #import pdb;pdb.set_trace()
            self.mask3d = mask3d.detach().cpu().numpy()
            
            for i,sub_out2d in enumerate(out2d_show):
                self.save_img_func(self.applyColor(sub_out2d.cpu().numpy()[0]).transpose(2, 0, 1), self.img_path,
                                   f'/out2d_{i}',dim=3)
            for i,sub_out3d in enumerate(out3d_show):
                self.save_img_func_3d(sub_out3d.detach().cpu().numpy()[0][0], self.img_path, f'/out3d_{i}')
        self.batch_idx = batch_idx

    def on_test_epoch_start(self):
        self.IoUMetric = IoUMetric(num_classes=3)

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
    def save_img_func_3d(self, img, img_path, save_dir=None, size=(1024, 1024)):
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
        cv2.imwrite(os.path.join(save_dir, img_name), color_img)
    def applyColor(self, pred):

        pred = np.stack([pred, pred, pred], 2)
        pred = np.uint8(pred)

        color_list = [[130, 217, 178], [191, 158, 142], [191, 158, 142]]  # [250,137,137]]

        # generate color for pred and gt
        for cls_id, color in enumerate(color_list):
            for cid, cvalue in enumerate(color):
                pred[:, :, cid][pred[:, :, cid] == cls_id + 1] = cvalue

        return pred
    def vis_3d(self, img, custom_lut, mM=None):

        if mM is None:
            m = img.min()#27.29
            M = img.max()#83.26 
        else:
            m, M = mM     

        img_gray = np.uint8(255 * self.norm(img, M, m))
        img_gray = np.stack([img_gray, img_gray, img_gray], axis=2)

        img_color = cv2.LUT(img_gray, custom_lut)
        return img_color

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
            dst.write(img)#, indexes=3)

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


