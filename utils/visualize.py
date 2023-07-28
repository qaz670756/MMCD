import cv2
import tifffile as tiff
import numpy as np
import os
import matplotlib.pyplot as plt

def norm(x, M, m):
    
    return (x-m)/(M-m+1e-6)

def vis_3d(img_path, custom_lut):
    if not os.path.isfile(img_path):
        return np.ones((500,500),dtype=float)
    img = tiff.imread(img_path)

    m = img.min()#27.29
    M = img.max()#83.26
    #print(img.max(),img.min(),img.shape)
    img_gray = np.uint8(255*norm(img,M,m))
    img_gray = np.stack([img_gray,img_gray,img_gray],axis=2)
    
    #print(img_gray.max(),img_gray.min())
    
    img_color = cv2.LUT(img_gray, custom_lut)#cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    #return cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
    return img_color

def genColormap():
    custom_lut = []#np.zeros((256, 1, 3), dtype=np.uint8)
    def convert_line(line):
        return [int(x) for x in line.split(',')[1:4]]
    lines = open('./utils/colormap.txt','r').readlines()
    for idx in range(len(lines)-1):
        lcolor = convert_line(lines[idx])
        rcolor = convert_line(lines[idx+1])
        if idx == 0:
            custom_lut.append(lcolor)
        
        R = np.linspace(lcolor[0],rcolor[0],6,dtype=int)[1:]
        G = np.linspace(lcolor[1],rcolor[1],6,dtype=int)[1:]
        B = np.linspace(lcolor[2],rcolor[2],6,dtype=int)[1:]
        
        for r,g,b in zip(R,G,B):
            custom_lut.append([r,g,b])

    #import pdb;pdb.set_trace()
    return np.array(custom_lut,dtype=np.uint8).reshape(256,1,3)
    #for 
 
def test():
    # Define a grayscale image
    gray_image = np.ones((256, 256,3), dtype=np.uint8)
    for i in range(256):
        gray_image[:,i,:] *= i
    # Define a custom LUT
    
    custom_lut = genColormap()

    # Apply the custom LUT to the grayscale image
    print(custom_lut.shape)
    color_image = cv2.LUT(gray_image, custom_lut)
    
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    img = tiff.imread(r'../data/amsterdam/test/3D/amsterdam_18_03.tif')
    cv2.imwrite('./utils/amsterdam_18_03.png',vis_3d(img, custom_lut))
    # Display the color image
    cv2.imwrite("./utils/ColorMap.png", color_image)

def extract_gtNames(gt_paths):
    names_2d = {}
    names_3d = {}
    for gt_path in gt_paths:
        city = gt_path.split('/')[2]
        with_change_samples = []
        for x in os.listdir(gt_path+'/2D'):
            if not x.endswith('.tif'):
                continue
            img_path = os.path.join(gt_path,'2D',x)
            #import pdb;pdb.set_trace()
            change_points = tiff.imread(img_path).sum()
            if change_points>0:
                with_change_samples.append([img_path,change_points])
        with_change_samples.sort(key=lambda x:-x[1])
        names_2d[city] = with_change_samples
        names_3d[city] = [[x[0].replace('2D','3D'),change_points] for x in with_change_samples]
    
    
    return names_2d,names_3d

def extract_imgs(models, gt_paths, rows=2):
    # here the rows means showing image count per city
    # img_dict {city_name: [
    #                        [gt_2d,pred1,pred2,pred_k], # row_0
    #                         ....
    #                       [gt_2d,pred1,pred2,pred_k] # row_idx-1   
    #           ]}
    names_2d,names_3d = extract_gtNames(gt_paths)
    cities = names_2d.keys()
    
    img_dict_2d = {x:[] for x in cities}
    img_dict_3d = {x:[] for x in cities}
    
    for city in cities:
        for row_idx in range(rows):
            gt_2d = names_2d[city][row_idx][0]
            gt_3d = names_3d[city][row_idx][0]
            img_name = gt_2d.split('/')[-1]
            tmp_2d = [gt_2d]
            tmp_3d = [gt_3d]
            for model in models:    
                path_2d = os.path.join('results',model,'lightning_logs','out2d',img_name)
                path_3d = os.path.join('results',model,'lightning_logs','out3d',img_name)
                tmp_2d.append(path_2d)
                tmp_3d.append(path_3d)
            img_dict_2d[city].append(tmp_2d)
            img_dict_3d[city].append(tmp_3d)

        
    return img_dict_2d, img_dict_3d
        
def show(img_dict, names=[], save_dir='./utils/change_3d.png',show_3d=False):
    print('Showing image...')
    custom_lut = genColormap()
    cities = list(img_dict.keys())
    rows_per_city = len(img_dict[cities[0]])
    rows = len(cities)*rows_per_city
    cols = len(img_dict[cities[0]][-1])
    f, ax = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    plt.tight_layout()
    plt.subplots_adjust(wspace=1,hspace=1)
    f.tight_layout()
    for cid, city in enumerate(cities):
        for row_id in range(len(img_dict[city])):
            for col_id in range(cols):
                #import pdb;pdb.set_trace()
                #try:
                if show_3d:
                    image = vis_3d(img_dict[city][row_id][col_id],custom_lut)
                else:
                    if not os.path.isfile(img_dict[city][row_id][col_id]):
                        image = np.ones((500,500),dtype=float)
                    else:
                        image = tiff.imread(img_dict[city][row_id][col_id])
                #except:
                #    image = np.ones((512,512),dtype=np.uint8)
                ax[row_id+cid*rows_per_city, col_id].imshow(image,cmap='gray')
                ax[row_id+cid*rows_per_city, col_id].set_xticks([])
                ax[row_id+cid*rows_per_city, col_id].set_yticks([])
    
    for _ in range(cols):
        ax[row_id+cid*rows_per_city, _].set_xlabel(names[_], fontsize=12)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_dir, dpi=300, pad_inches=0.05, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    #test()
    models = ['CD_siamunet_ef','CD_siamunet_diff','CD_siamunet_conc','CD_ifn_deep3d',
              'CD_ori_range_snunet','CD_changeformer_emb128','CD_p2vnet','CD_hmcdnet']
    names = ['GT', 'FC-EF','FC-diff','FC-conc','IFN_deep3d',
              'SNUNet','Changeformer','P2VNet','MTBIT']
    gt_paths = ['../data/amsterdam/test','../data/rotterdam/test','../data/utrecht/test']
    img_dict_2d, img_dict_3d = extract_imgs(models, gt_paths, rows=2)
    show(img_dict_3d, names=names, save_dir='./utils/change_3d.png',show_3d=True)
    show(img_dict_2d, names=names, save_dir='./utils/change_2d.png',show_3d=False)
    #import pdb;pdb.set_trace()



