import os
import tifffile as tiff
import rasterio as ro
from tqdm import tqdm
import numpy as np
import cv2


def genColormap():
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
    return np.array(custom_lut, dtype=np.uint8).reshape(256, 1, 3)

def norm(x, M, m):
    return (x - m) / (M - m)
def vis_3d(img, custom_lut):

    m = img.min()  # 27.29
    M = img.max()  # 83.26
    # print(img.max(),img.min(),img.shape)
    img_gray = np.uint8(255 * norm(img, M, m))
    img_gray = np.stack([img_gray, img_gray, img_gray], axis=2)

    img_color = cv2.LUT(img_gray, custom_lut)  # cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    # return cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
    return img_color

def save_img_func(img, img_path, save_dir=None, size=(1000,1000),dim=1):
    #if not save_dir:

    os.makedirs(save_dir, exist_ok=True)
    #img = cv2.resize(img,size)
    img_name = img_path.split('/')[-1]
    color_img = vis_3d(img, custom_lut)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, img_name),color_img)


def colorGT(root):
    names = os.listdir(root)
    img_list = [os.path.join(root, x) for x in names]

    for img_path in tqdm(img_list):
        img = tiff.imread(img_path)
        if img.sum()>0:
            save_img_func(img,img_path,'/home/liub/MMCD/utils/gt3D',dim=3)

if __name__ == '__main__':
    roots = ['/home/liub/data/amsterdam_025/test/mask3d_1k',
             '/home/liub/data/rotterdam_025/test/mask3d_1k',
             '/home/liub/data/utrecht_025/test/mask3d_1k']
    custom_lut = genColormap()
    for root in roots:
        colorGT(root)
