import os
import numpy as np
import tifffile as tiff
import skimage
import imageio as iio

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu


def center_crop(imm, size, imtype='image'):
    h = int(size[0] / 2)
    w = int(size[1] / 2)
    ch = int(imm.shape[0] / 2)
    cw = int(imm.shape[1] / 2)
    if imtype == 'image':
        return imm[ch - h:ch + h, cw - w:cw + w, :]
    else:
        return imm[ch - h:ch + h, cw - w:cw + w]


def extract_names(root, sets):
    if len(sets) == 5:
        t1_images_dir = os.path.join(root, sets[0])
        t2_images_dir = os.path.join(root, sets[1])
        masks2d_dir = os.path.join(root, sets[2])
        masks3d_dir = os.path.join(root, sets[3])
        height_dir = os.path.join(root, sets[4])
    elif len(sets) == 4:
        t1_images_dir = os.path.join(root, sets[0])
        t2_images_dir = os.path.join(root, sets[1])
        masks2d_dir = os.path.join(root, sets[2])
        masks3d_dir = os.path.join(root, sets[3])

    ids = os.listdir(t1_images_dir)

    # important!
    ids = [x for x in ids if x.endswith('.tif')]

    t1_images_fps = [os.path.join(t1_images_dir, image_id) for image_id in ids]
    t2_images_fps = [os.path.join(t2_images_dir, image_id) for image_id in ids]
    masks2d_fps = [os.path.join(masks2d_dir, image_id) for image_id in ids]
    masks3d_fps = [os.path.join(masks3d_dir, image_id) for image_id in ids]

    if len(sets) == 5:
        height_fps = [os.path.join(height_dir, image_id) for image_id in ids]
        return t1_images_fps, t2_images_fps, masks2d_fps, masks3d_fps, height_fps
    else:
        return t1_images_fps, t2_images_fps, masks2d_fps, masks3d_fps


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    """

    def __init__(
            self,
            roots,
            sets=[],
            augmentation=False,
            return_crs=False
    ):
        if type(roots) != list:
            roots = [roots]
        self.t1_images_fps = []
        self.t2_images_fps = []
        self.masks2d_fps = []
        self.masks3d_fps = []
        self.height_fps = []

        for root in roots:
            if len(sets)==5:
                r1, r2, r3, r4, r5 = extract_names(root, sets)
                self.t1_images_fps += r1
                self.t2_images_fps += r2
                self.masks2d_fps += r3
                self.masks3d_fps += r4
                self.height_fps += r5
            else:
                r1, r2, r3, r4 = extract_names(root, sets)
                self.t1_images_fps += r1
                self.t2_images_fps += r2
                self.masks2d_fps += r3
                self.masks3d_fps += r4


        self.augmentation = augmentation
        self.return_crs = return_crs
        self.sets_len = len(sets)

        if len(sets)==5:
            self.check([self.t1_images_fps,self.t2_images_fps,self.masks2d_fps,
                        self.masks3d_fps,self.height_fps])
        else:
            self.check([self.t1_images_fps,self.t2_images_fps,self.masks2d_fps,
                        self.masks3d_fps])
        #self.filter()

    def check(self, name_list):
        def name(s):
            return s.split('/')[-1]

        if self.sets_len==5:
            for r1,r2,r3,r4,r5 in zip(*name_list):
                if name(r1)==name(r2)==name(r3)==name(r4)==name(r5):
                    continue
                else:
                    print('Error!!!!!!!!!!!!!!')
        if self.sets_len==4:
            for r1,r2,r3,r4 in zip(*name_list):
                if name(r1)==name(r2)==name(r3)==name(r4):
                    continue
                else:
                    print('Error!!!!!!!!!!!!!!')
    def filter(self):
        for name in self.masks2d_fps:
            mask2d = iio.imread(name)
            if mask2d.sum()==0:
                idx = self.masks2d_fps.index(name)
                del(self.t1_images_fps[idx])
                del(self.t2_images_fps[idx])
                del(self.masks2d_fps[idx])
                del(self.masks3d_fps[idx])

    def rescale(self, img):
        
        def norm(x, M, m):
            return (x-m)/(M-m)
        
        M = 40
        m = -10

        img = norm(img,M,m)
        img = img * 255.0

        return img
    
    def __getitem__(self, i):

        # read data with tifffile because of 3d mask int16
        t1 = iio.imread(self.t1_images_fps[i])[:, :, :3]  # .transpose([2,0,1])
        t2 = iio.imread(self.t2_images_fps[i])  # [:,:,:3]#.transpose([2,0,1])
        t2 = self.rescale(t2)
        if t2.shape[0]!=3:
            t2 = np.stack([t2, t2, t2], axis=2)
            
        #import pdb;pdb.set_trace()
        
        mask2d = iio.imread(self.masks2d_fps[i])
        mask3d = tiff.imread(self.masks3d_fps[i])
        
        
        if self.sets_len == 5:
            height = tiff.imread(self.height_fps[i])

        # apply augmentations
        if self.augmentation:
            t1 = np.uint8(t1)
            if len(t2.shape)==3:
                t2 = np.uint8(t2)
            #import pdb;pdb.set_trace()
            if self.sets_len == 5:
                sample = self.augmentation(image=t1, t2=t2, mask=mask2d, mask3d=mask3d, height=height)
                
                t1, t2, mask2d, mask3d, height = sample['image'], sample['t2'], sample['mask'], sample['mask3d'], sample['height']
 
            else:
                sample = self.augmentation(image=t1, t2=t2, mask=mask2d, mask3d=mask3d)
                t1, t2, mask2d, mask3d = sample['image'], sample['t2'], sample['mask'], sample['mask3d']
            


        if self.return_crs:
            if self.sets_len == 5:
                return t1, t2, mask2d, mask3d.float(), height.float(), self.t1_images_fps[i]
            else:
                return t1, t2, mask2d, mask3d.float(), self.t1_images_fps[i]
        else:
            if self.sets_len == 5:
                return t1, t2, mask2d, mask3d.float(), height.float()
            else:
                return t1, t2, mask2d, mask3d.float()

    def __len__(self):
        return len(self.t1_images_fps)
