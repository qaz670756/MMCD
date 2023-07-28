import os
import tifffile as tiff
import rasterio as ro
from tqdm import tqdm
import numpy as np
import cv2


def interGT(root):
    root += 'mask2d_1k/'
    names = os.listdir(root)
    img_list_2d = [os.path.join(root, x) for x in names]
    img_list_3d = [x.replace('mask2d_1k','mask3d_1k') for x in img_list_2d]
    os.makedirs('/home/liub/MMCD/utils/interGT/',exist_ok=True)
    for img_path_2d, img_path_3d in tqdm(zip(img_list_2d,img_list_3d)):
        img_2d = tiff.imread(img_path_2d)
        img_3d = tiff.imread(img_path_3d)
        pesudo_2d = img_3d.copy()
        pesudo_2d[pesudo_2d>0]=2
        pesudo_2d[pesudo_2d<0]=1
        if img_2d.sum()>0:
            #import pdb;pdb.set_trace()
            interaction = pesudo_2d==img_2d
            interaction *= pesudo_2d>0
            interaction = np.uint8(interaction*255)

            cv2.imwrite('/home/liub/MMCD/utils/interGT/'+img_path_2d.split('/')[-1],interaction)

if __name__ == '__main__':
    roots = ['/home/liub/data/amsterdam_025/test/',
             '/home/liub/data/rotterdam_025/test/',
             '/home/liub/data/utrecht_025/test/']
    
    for root in roots:
        interGT(root)
