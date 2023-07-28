import numpy as np
import numpy.ma as ma
from ssim import ssim

eps = 1e-10
def get_mask_array(inputs, targets, mask):
    mask_ = mask.copy()
    indices_one = mask_ == 1
    indices_zero = mask_ == 0
    mask_[indices_one] = 0  # replacing 1s with 0s
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
