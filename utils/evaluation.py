import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

def getHist(img,num_bins):
    hist, bins = np.histogram(img, bins=np.linspace(-25, 70,num_bins+1))
    return hist, bins

def drawFig(hist1,hist2,bins,img_path,num_bins):
    f, ax = plt.subplots()
    plt.tight_layout()
    #import pdb;pdb.set_trace()
    ax.bar(bins[:-1], hist1, width=110/num_bins, color='blue', alpha=0.8)
    ax.bar(bins[:-1], hist2, width=110/num_bins, color='yellow', alpha=0.8)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Accumulated Pixel Value Distribution")
    ax.set_ylim([0, 300000])
    f.savefig(img_path.replace('png','svg'), dpi=300, pad_inches=0.05,
              bbox_inches='tight', format='svg')

def accHist(root):
    total_hist = np.zeros(110)
    img_list = os.listdir(root)
    for img_name in img_list:
        img_path = os.path.join(root, img_name)
        img = tiff.imread(img_path)
        mask = tiff.imread(img_path.replace('3d','2d'))
        if mask.sum()>0:
            hist, bins = getHist(img[mask!=0])
            total_hist += hist

    return total_hist, bins

if __name__ == '__main__':
    total_hist = np.zeros(110)
    roots = ['/home/liub/data/amsterdam_025/test/mask3d_1k',
             '/home/liub/data/rotterdam_025/test/mask3d_1k',
             '/home/liub/data/utrecht_025/test/mask3d_1k']
    
    for root in roots:
        hist, bins = accHist(root)
        total_hist += hist
    f, ax = plt.subplots()
    plt.tight_layout()

    ax.bar(bins[:-1], total_hist, width=1)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Accumulated Pixel Value Distribution")
    f.savefig('./utils/GTdistribution.png', dpi=300, pad_inches=0.05, bbox_inches='tight')
