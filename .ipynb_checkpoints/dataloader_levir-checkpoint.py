import os
import numpy as np
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
        
        

        self.augmentation = augmentation
        self.return_crs = return_crs
        self.sets_len = len(sets)
        
        ids = os.listdir(os.path.join(roots,sets[0]))
        ids = [x for x in ids if x.endswith('.png')]
        
        self.t1_images_fps = [os.path.join(roots,sets[0],x) for x in ids]
        self.t2_images_fps = [os.path.join(roots,sets[1],x) for x in ids]
        self.labels_fps = [os.path.join(roots,sets[2],x) for x in ids]


    def __getitem__(self, i):

        # read data with tifffile because of 3d mask int16
        t1 = iio.imread(self.t1_images_fps[i])[:, :, :3]  # .transpose([2,0,1])
        t2 = iio.imread(self.t2_images_fps[i])[:, :, :3]  # [:,:,:3]#.transpose([2,0,1])
        mask = iio.imread(self.labels_fps[i])
        mask[mask!=0] = 1
        #import pdb;pdb.set_trace()
        if self.augmentation:
            
            t1 = np.uint8(t1)
            t2 = np.uint8(t2)

            sample = self.augmentation(image=t1, t2=t2, mask=mask)
            t1, t2, mask= sample['image'], sample['t2'], sample['mask']
            

        if self.return_crs:
            return t1, t2, mask, self.t1_images_fps[i]

        else:
            return t1, t2, mask

    def __len__(self):
        return len(self.t1_images_fps)
