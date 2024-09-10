import numpy as np
import numpy.ma as ma
#from utils.chamfer_3D.dist_chamfer_3D import chamfer_3DDist_nograd as CD
import torch
#from ssim import ssim

eps = 1e-10
#cd = CD()
def get_mask_array(inputs, targets, mask, thresh=1e-3):
    # The thresh is used for excluding really small height values that less than 0.001 m
    # since the small non-zero values like 1e-5 would cause uncorrect Rel output
    mask_ = mask.copy()
    indices_one = mask_ == 1
    indices_zero = mask_ == 0
    mask_[indices_one] = 0  # replacing 1s with 0s
    mask_[np.abs(targets)<thresh] = 1
    mask_[indices_zero] = 1  # replacing 0s with 1s
    inputs = ma.masked_array(inputs, mask=mask_)
    targets = ma.masked_array(targets, mask=mask_)

    return inputs, targets

# to calculate rmse
def metric_mse(inputs, targets, mask, exclude_zeros = False):
    if exclude_zeros:
        if mask.sum()!=0:
            inputs, targets = get_mask_array(inputs, targets, mask)
            loss = (inputs - targets) ** 2
            n_pixels = np.count_nonzero(targets)
            #import pdb;pdb.set_trace()
            #n_pixels = 1 if n_pixels==0 else n_pixels
            return np.sum(loss)/n_pixels
        else:
            return 0.0
    else:
        loss = (inputs - targets) ** 2

        return np.mean(loss)

def metric_rel(inputs, targets, mask):
    
    if targets.sum()==0:
        return 0
    
    inputs, targets = get_mask_array(inputs, targets, mask)
    result = (inputs-targets)/(targets+eps)
    #if np.mean(np.abs(result))>5:
    #    import pdb;pdb.set_trace()
    return np.mean(np.abs(result))

def metric_rellog10(inputs, targets, mask):
    if targets.sum()==0:
        return 0
    inputs, targets = get_mask_array(inputs, targets, mask)
    result = np.log10(inputs+eps) - np.log10(targets+eps, where=(mask!=0))

    return np.mean(np.abs(result))

def metric_ncc(inputs, targets, mask):

    if targets.sum()==0:
        return 0
    #inputs, targets = get_mask_array(inputs, targets, mask)
    mean_He, mean_Hr = inputs.mean(), targets.mean()
    std_He, std_Hr = np.std(inputs)+eps, np.std(targets)+eps
    ncc = (inputs-mean_He)*(targets-mean_Hr)/(std_He*std_Hr)

    return np.mean(ncc)

def metric_ssim(inputs, targets):
    return ssim(inputs.cuda(), targets.cuda())




class metric_chamferDist(torch.nn.Module):
    def __init__(self,H=512,W=512,grid_size=128,res=0.25):
        super(metric_chamferDist, self).__init__()
        self.grid_x, self.grid_y = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
        self.grid_x, self.grid_y = self.grid_x.ravel()*res, self.grid_y.ravel()*res
        self.grid_size = grid_size
        
        #grid_x, grid_y = grid_x.to(inputs.device), grid_y.to(inputs.device)
    def func(self, inputs, targets):
        self.grid_x, self.grid_y = self.grid_x.to(inputs.device), self.grid_y.to(inputs.device)
        #import pdb;pdb.set_trace()
        input_grids = inputs[0][0].unfold(0, self.grid_size, self.grid_size).unfold(1, self.grid_size, self.grid_size)
        input_grids = inputs.reshape(-1,self.grid_size, self.grid_size)
        target_grids = targets[0].unfold(0, self.grid_size, self.grid_size).unfold(1, self.grid_size, self.grid_size)
        target_grids = targets.reshape(-1,self.grid_size, self.grid_size)
        dist = []
        for grid_pred, grid_gt in zip(input_grids, target_grids):
            pred = torch.stack((self.grid_x, self.grid_y,grid_pred.ravel()),0).transpose(1,0).unsqueeze(0)
            gt = torch.stack((self.grid_x, self.grid_y,grid_gt.ravel()),0).transpose(1,0).unsqueeze(0)
        
            dist1, dist2, idx1, idx2 = cd(gt,pred)
            dist.append((dist1.sum()+dist2.sum()).detach().cpu().numpy())
        #import pdb;pdb.set_trace()

        return np.mean(dist)