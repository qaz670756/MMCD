
import lightning.pytorch as pl
import rasterio as ro, logging
import numpy as np
from torch import optim, nn, utils, Tensor
from .optim import set_scheduler
import os
import cv2

class pl_trainer_base(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def norm(self, x, M, m):
        return (x - m) / (M - m)

    def reverse_norm(self, x, M, m):
        return x * (M - m) + m

    def norm_train(self, x):

        return 2 * (x - self.min_scale) / (self.max_scale - self.min_scale) - 1

    def norm_infer(self, x):

        return (x + 1) * (self.max_scale - self.min_scale) / 2 + self.min_scale
    
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
    
    def save_img_func(self, img, img_path, save_dir=None, size=(1024, 1024), dim=1):
        # if not save_dir:
        save_dir = self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]) + save_dir
        os.makedirs(save_dir, exist_ok=True)
        # img = cv2.resize(img,size)
        img_name = img_path[0].split('/')[-1]
        src = ro.open(img_path[0])
        with ro.open(os.path.join(save_dir, img_name), mode='w', driver='GTiff',
                     width=size[0], height=size[1],
                     count=dim, crs=src.crs, transform=src.transform, dtype=img.dtype) as dst:  #
            # import pdb;pdb.set_trace()
            dst.write(img)  # , indexes=3)

    def save_img_func_3d_backup(self, img, img_path, save_dir=None, size=(1024, 1024)):
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
        #color_img = self.vis_3d(img,self.custom_lut, (self.mask3d.min(), self.mask3d.max()))
        color_img = self.vis_3d(img, self.custom_lut, (self.min_scale+10, self.max_scale-60))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, img_name), color_img)

    def save_img_func_3d(self, img, img_path, save_dir=None, size=(1024, 1024)):
        # if not save_dir:
        save_dir = self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]) + save_dir
        os.makedirs(save_dir, exist_ok=True)
        # img = cv2.resize(img,size)
        img_name = img_path[0].split('/')[-1]
        src = ro.open(img_path[0])
        # with ro.open(os.path.join(save_dir, img_name), mode='w', driver='GTiff',
        #              width=size[0], height=size[1],
        #              count=1, crs=src.crs, transform=src.transform, dtype=img.dtype) as dst:  #
        #     dst.write(img.reshape(1,1024,1024))
        #color_img = self.vis_3d(img,self.custom_lut, (self.mask3d.min(), self.mask3d.max()))
        color_img = self.vis_3d(img, self.custom_lut, (self.min_scale+10, self.max_scale-60))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, img_name), color_img)

    def save_vis_img_func(self,vis_feature,img_path,save_dir):
        save_dir = self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]) + save_dir
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, img_path[0].split('/')[-1]), vis_feature)

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



    def applyColor(self, pred):

        pred = np.stack([pred, pred, pred], 2)
        pred = np.uint8(pred)

        color_list = [[130, 217, 178], [191, 158, 142], [191, 158, 142]]  # [250,137,137]]

        # generate color for pred and gt
        for cls_id, color in enumerate(color_list):
            for cid, cvalue in enumerate(color):
                pred[:, :, cid][pred[:, :, cid] == cls_id + 1] = cvalue

        return pred


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