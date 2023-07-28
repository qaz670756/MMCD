import os
import tifffile as tiff
import rasterio as ro
from tqdm import tqdm
import numpy as np
import cv2


def save_img_func(img, img_path, save_dir=None, size=(1000,1000),dim=3):
    #if not save_dir:

    os.makedirs(save_dir, exist_ok=True)
    #img = cv2.resize(img,size)
    img_name = img_path.split('/')[-1]
    src = ro.open(img_path)
    with ro.open(os.path.join(save_dir, img_name), mode='w', driver='GTiff', 
                    width=size[0], height=size[1],
                    count=dim, crs=src.crs, transform=src.transform, dtype=img.dtype) as dst:  #
        #import pdb;pdb.set_trace()
        dst.write(img.transpose(2,0,1))#, indexes=3)

def applyColor(pred):

    pred = np.stack([pred, pred, pred], 2)
    pred = np.uint8(pred)

    color_list = [[130,217,178],[191,158,142], [191,158,142]]

    # generate color for pred and gt
    for cls_id, color in enumerate(color_list):
        for cid, cvalue in enumerate(color):
            pred[:, :, cid][pred[:, :, cid] == cls_id + 1] = cvalue
    
    return pred


def binaryGT(root):
    names = os.listdir(root)
    img_list = [os.path.join(root, x) for x in names]

    for img_path in tqdm(img_list):
        img = tiff.imread(img_path)
        img[img>0] = 2
        img[img<0] = 1
        if img.sum()>0:
            img = applyColor(img)
            save_img_func(img,img_path,'/home/liub/MMCD/utils/gt3D_semantic')

if __name__ == '__main__':
    roots = ['/home/liub/data/amsterdam_025/test/mask3d_1k',
             '/home/liub/data/rotterdam_025/test/mask3d_1k',
             '/home/liub/data/utrecht_025/test/mask3d_1k']

    for root in roots:
        binaryGT(root)
