from __future__ import print_function
import numpy as np
import numpy.ma as ma
import json
import time
import sys
from datetime import datetime
import pathlib
import shutil
import yaml
from argparse import ArgumentParser
import os
from functools import partial
from sklearn import metrics
from tqdm import tqdm, trange
import torchvision.models as models

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from models.SUNet18 import SUNet18
from models.MTBIT import MTBIT, MTBIT_par_HE
from dataloader import Dataset
from augmentations import get_validation_augmentations_he, get_training_augmentations_he
from losses import choose_criterion3d, choose_criterion2d
from optim import set_optimizer, set_scheduler
from cp import pretrain_strategy


def get_args():
    parser = ArgumentParser(description="Hyperparameters", add_help=True)
    parser.add_argument('-c', '--config-name', type=str, help='YAML Config name', dest='CONFIG', default='MTBIT')
    parser.add_argument('-nw', '--num-workers', type=str, help='Number of workers', dest='num_workers', default=0)
    parser.add_argument('-v', '--verbose', type=bool, help='Verbose validation metrics', dest='verbose', default=False)
    return parser.parse_args()


# to calculate rmse
def metric_mse(inputs, targets, mask, exclude_zeros=False):
    if exclude_zeros:
        if mask.sum() != 0:
            mask_ = mask.copy()
            indices_one = mask_ == 1
            indices_zero = mask_ == 0
            mask_[indices_one] = 0  # replacing 1s with 0s
            mask_[indices_zero] = 1  # replacing 0s with 1s
            inputs = ma.masked_array(inputs, mask=mask_)
            targets = ma.masked_array(targets, mask=mask_)
            loss = (inputs - targets) ** 2
            n_pixels = np.count_nonzero(targets)
            # import pdb;pdb.set_trace()
            # n_pixels = 1 if n_pixels==0 else n_pixels
            return np.sum(loss) / n_pixels
        else:
            return 0.0
    else:
        loss = (inputs - targets) ** 2
        return np.mean(loss)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


args = get_args()

device = 'cuda'
cuda = True
num_GPU = 1
torch.cuda.set_device(0)
manual_seed = 18
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

config_name = args.CONFIG
config_path = './config/' + config_name
default_dst_dir = "./results/"
out_file = default_dst_dir + config_name + '/'
os.makedirs(out_file, exist_ok=True)

# Load the configuration params of the experiment
full_config_path = config_path + ".yaml"
print(f"Loading experiment {full_config_path}")
with open(full_config_path, "r") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

print(f"Logs and/or checkpoints will be stored on {out_file}")
shutil.copyfile(full_config_path, out_file + 'config.yaml')
print("Config file correctly saved!")

stats_file = open(out_file + 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv), file=stats_file)
print(' '.join(sys.argv))

print(exp_config)
print(exp_config, file=stats_file)

x_train_dir = exp_config['data']['train']['path']
x_valid_dir = exp_config['data']['val']['path']
x_test_dir = exp_config['data']['test']['path']

batch_size = exp_config['data']['batch_size']

lweight2d, lweight3d = exp_config['model']['loss_weights']


augmentation = exp_config['data']['augmentations']
min_scale = exp_config['data']['min_value']
max_scale = exp_config['data']['max_value']

mean = exp_config['data']['mean']
std = exp_config['data']['std']

if augmentation:
    train_transform = get_training_augmentations_he(m=mean, s=std)
else:
    train_transform = get_validation_augmentations_he(m=mean, s=std)

valid_transform = get_validation_augmentations_he(m=mean, s=std)

train_dataset = Dataset(x_train_dir, exp_config['data']['sets'],
                        augmentation=train_transform)

valid_dataset = Dataset(x_valid_dir, exp_config['data']['sets'],
                        augmentation=valid_transform)

test_dataset = Dataset(x_test_dir, exp_config['data']['sets'],
                       augmentation=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

name_3dloss = exp_config['model']['3d_loss']
exclude_zeros = exp_config['model']['exclude_zeros']
criterion3d = choose_criterion3d(name=name_3dloss)

weights2d = exp_config['model']['2d_loss_weights']
class_weights2d = torch.FloatTensor(weights2d).to(device)
name_2dloss = exp_config['model']['2d_loss']
criterion2d = choose_criterion2d(name_2dloss, class_weights2d)  # , class_ignored)

nepochs = exp_config['optim']['num_epochs']
lr = exp_config['optim']['lr']

model = exp_config['model']['model']
classes = exp_config['model']['num_classes']

pretrain = exp_config['model']['pretraining_strategy']
arch = exp_config['model']['feature_extractor_arch']
CHECKPOINTS = exp_config['model']['checkpoints_path']

encoder, pretrained, _ = pretrain_strategy(pretrain, CHECKPOINTS, arch)

try:
    sigmoid3d = exp_config['model']['sigmoid3d']
except:
    sigmoid3d = False
print('sigmoid3d=', sigmoid3d)
if model == "SUNet18":
    net = SUNet18(3, 2, resnet=encoder).to(device)
elif model == 'mtbit_resnet18':
    net = MTBIT_par_HE(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True, sigmoid3d=sigmoid3d,
                enc_depth=1, dec_depth=8, decoder_dim_head=16).to(device)
elif model == 'mtbit_resnet50':
    net = MTBIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True, sigmoid3d=sigmoid3d,
                with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=16, backbone='resnet50').to(device)
else:
    print('Model not implemented yet')

print('Model selected: ', model)

optimizer = set_optimizer(exp_config['optim'], net)
print('Optimizer selected: ', exp_config['optim']['optim_type'])
lr_adjust = set_scheduler(exp_config['optim'], optimizer)
print('Scheduler selected: ', exp_config['optim']['lr_schedule_type'])

res_cp = exp_config['model']['restore_checkpoints']
#if os.path.exists(out_file + f'{res_cp}bestnet.pth'):
#    net.load_state_dict(torch.load(out_file + f'{res_cp}bestnet.pth'))
#    print('Checkpoints successfully loaded!')
#else:
#    print('No checkpoints founded')

tr_par, tot_par = count_parameters(net)
print(f'Trainable parameters: {tr_par}, total parameters {tot_par}')
print(f'Trainable parameters: {tr_par}, total parameters {tot_par}', file=stats_file)

import pdb;pdb.set_trace()

start = time.time()

best2dmetric = 0
best3dmetric = 1000000

net.train()

for epoch in range(1, nepochs):
    tot_2d_loss = 0
    tot_3d_loss = 0

    for param_group in optimizer.param_groups:
        print("Epoch: %s" % epoch, " - Learning rate: ", param_group['lr'])

    for t1, t2, mask2d, mask3d, maskheight in tqdm(train_loader):

        t1 = t1.to(device)
        t2 = t2.to(device)

        mask3d = mask3d.to(device).float()
        maskheight = maskheight.to(device).float()
        out2d, out3d, outheight = net(t1, t2)

        mask_weight_3d = mask2d.clone().detach().cuda()*0.95
        mask_weight_3d[mask_weight_3d==0] = 0.05

        if args.verbose:
            print()
            print('MASK 3D: ', torch.min(mask3d).item(), torch.max(mask3d).item())

        if sigmoid3d:
            mask3d = (mask3d - min_scale) / (max_scale - min_scale)
        else:
            mask3d = 2 * (mask3d - min_scale) / (max_scale - min_scale) - 1
            maskheight = 2 * (maskheight - min_scale) / (max_scale - min_scale) - 1

        if args.verbose:
            print('MASK 3D NORM: ', torch.min(mask3d).item(), torch.max(mask3d).item())
            print('OUT 3D: ', torch.min(out3d).item(), torch.max(out3d).item())

        loss2d = criterion2d(out2d, mask2d.to(device).long())  # long
        #loss3d = criterion3d(out3d.squeeze(dim=1), mask3d, weights=mask_weight_3d)  # , exclude_zeros = exclude_zeros)
        loss3d = criterion3d(out3d.squeeze(dim=1), mask3d)
        loss3d += criterion3d(outheight.squeeze(dim=1), maskheight)

        loss = lweight2d * loss2d + lweight3d * loss3d  # sommo le loss

        optimizer.zero_grad()
        loss.backward()  # backward delle loss
        optimizer.step()

        tot_2d_loss += loss2d.detach().cpu().numpy() * batch_size
        tot_3d_loss += loss3d.detach().cpu().numpy() * batch_size

    # import pdb;pdb.set_trace()
    epoch_2d_loss = lweight2d * tot_2d_loss / len(train_dataset)
    epoch_3d_loss = lweight3d * tot_3d_loss / len(train_dataset)
    epoch_loss = lweight2d * epoch_2d_loss + lweight3d * epoch_3d_loss

    lr_adjust.step()

    print(f"Training loss: {epoch_loss},\t2D Loss: {epoch_2d_loss}, \t3D Loss: {epoch_3d_loss}")

    if epoch%40==0:
        with torch.no_grad():
            net.eval()

            TN = 0
            FP = 0
            FN = 0
            TP = 0
            mean_mae = 0
            mean_mae_height = 0
            rmse1 = 0
            rmse2 = 0
            rmse1_height = 0
            rmse2_height = 0

            for t1, t2, mask2d, mask3d, maskheight in tqdm(valid_loader):

                t1 = t1.to(device)
                t2 = t2.to(device)

                out2d, out3d, outheight = net(t1, t2)
                out2d = out2d.detach().argmax(dim=1).cpu().numpy()
                out3d = out3d.detach().cpu().numpy()
                outheight = outheight.detach().cpu().numpy()

                if sigmoid3d:
                    out3d = out3d * (max_scale - min_scale) + min_scale
                else:
                    out3d = ((out3d.ravel() + 1) / 2) * (max_scale - min_scale) + min_scale
                    outheight = ((outheight.ravel() + 1) / 2) * (max_scale - min_scale) + min_scale

                try:
                    tn, fp, fn, tp = metrics.confusion_matrix(mask2d.ravel(), out2d.ravel()).ravel()
                except:
                    tn, fp, fn, tp = [0, 0, 0, 0]
                    #print('Only 0 mask')

                    # try:
                #    tn, fp, fn, tp = metrics.confusion_matrix(mask2d.ravel(), out2d.ravel()).ravel()
                # except:
                #    import pdb;pdb.set_trace()
                mean_ae = metrics.mean_absolute_error(mask3d.ravel(), out3d.ravel())


                s_rmse1 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(),
                                     exclude_zeros=False)
                s_rmse2 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(),
                                     exclude_zeros=True)
                mean_ae_height = metrics.mean_absolute_error(maskheight.ravel(), outheight.ravel())
                s_rmse1_height = metric_mse(outheight.ravel(), maskheight.cpu().numpy().ravel(),
                                            mask2d.cpu().numpy().ravel(), exclude_zeros=False)
                s_rmse2_height = metric_mse(outheight.ravel(), maskheight.cpu().numpy().ravel(),
                                            mask2d.cpu().numpy().ravel(), exclude_zeros=True)
                max_error = metrics.max_error(mask3d.ravel(), out3d.ravel())
                mask_max = np.abs(mask3d.cpu().numpy()).max()

                mean_mae += mean_ae
                mean_mae_height += mean_ae_height
                rmse1 += s_rmse1
                rmse2 += s_rmse2
                rmse1_height += s_rmse1_height
                rmse2_height += s_rmse2_height
                TN += tn
                FP += fp
                FN += fn
                TP += tp

            mean_mae = mean_mae / len(valid_loader)
            mean_mae_height = mean_mae_height / len(valid_loader)
            mIoU = TP / (TP + FN + FP)
            mean_f1 = 2 * TP / (2 * TP + FP + FN)
            RMSE1 = np.sqrt(rmse1 / len(valid_loader))
            RMSE2 = np.sqrt(rmse2 / len(valid_loader))
            RMSE1_height = np.sqrt(rmse1_height / len(valid_loader))
            RMSE2_height = np.sqrt(rmse2_height / len(valid_loader))

            print(
                f'Validation metrics - 2D: F1 Score -> {mean_f1 * 100} %; mIoU -> {mIoU * 100} %; 3D: '+
                f'MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m: MAE_height -> {mean_mae_height} m; '+
                f'RMSE_height -> {RMSE1_height} m; cRMSE_height  -> {RMSE2_height} m')

            if mean_f1 > best2dmetric:
                best2dmetric = mean_f1
            torch.save(net.state_dict(), out_file + '/2dbestnet.pth')
            print('Best 2D model saved!')

            if RMSE2 < best3dmetric:
                best3dmetric = RMSE2
            torch.save(net.state_dict(), out_file + '/3dbestnet.pth')
            print('Best 3D model saved!')

            stats = dict(epoch=epoch, Loss2D=epoch_2d_loss, Loss3D=epoch_3d_loss, Loss=epoch_loss, RMSE=RMSE1, cRMSE=RMSE2,
                         F1Score=mean_f1 * 100)
        print(json.dumps(stats), file=stats_file)

end = time.time()
print('Training completed. Program processed ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600,
      'h')
print(f'Best metrics: F1 score -> {best2dmetric * 100} %,\t cRMSE -> {best3dmetric}')


torch.save(net.state_dict(), out_file + '/lastbestnet.pth')


start = time.time()

if os.path.exists('%s/' % out_file + f'{res_cp}bestnet.pth'):
    net.load_state_dict(torch.load('%s/' % out_file + f'{res_cp}bestnet.pth'))
    print("Checkpoints correctly loaded: ", out_file)

net.eval()

TN = 0
FP = 0
FN = 0
TP = 0
mean_mae = 0
rmse1 = 0
rmse2 = 0

for t1, t2, mask2d, mask3d, maskheight in tqdm(test_loader):

    t1 = t1.to(device)
    t2 = t2.to(device)

    out2d, out3d, outheight = net(t1, t2)
    out2d = out2d.detach().argmax(dim=1)
    out2d = out2d.cpu().numpy()
    out3d = out3d.detach().cpu().numpy()
    outheight = outheight.detach().cpu().numpy()

    if sigmoid3d:
        out3d = out3d * (max_scale - min_scale) + min_scale
    else:
        out3d = (out3d + 1) * (max_scale - min_scale) / 2 + min_scale  # Tanh
        height3d = (height3d + 1) * (max_scale - min_scale) / 2 + min_scale

    try:
        tn, fp, fn, tp = metrics.confusion_matrix(mask2d.ravel(), out2d.ravel()).ravel()
    except:
        tn, fp, fn, tp = [0, 0, 0, 0]
        print('Only 0 mask')

    mean_ae = metrics.mean_absolute_error(mask3d.ravel(), out3d.ravel())
    s_rmse1 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros=False)
    s_rmse2 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros=True)
    max_error = metrics.max_error(mask3d.ravel(), out3d.ravel())
    mask_max = np.abs(mask3d.cpu().numpy()).max()

    if args.verbose:
        print()
        print(f'2D Val: TN: {tn},\tFN: {fn},\tTP: {tp},\tFP: {fp},\tF1 Score: {f1_score},\tIoU: {IoU}')
        print(
            f'3D Val: Mean Absolute Error: {mean_ae}, \tRMSE Error: {s_rmse}, \tMax Error: {max_error} (w.r.t {mask_max})')

    mean_mae += mean_ae
    rmse1 += s_rmse1
    rmse2 += s_rmse2
    TN += tn
    FP += fp
    FN += fn
    TP += tp

mean_mae = mean_mae / len(test_loader)
mean_f1 = 2 * TP / (2 * TP + FP + FN)
mIoU = TP / (TP + FN + FP)
RMSE1 = np.sqrt(rmse1 / len(test_loader))
RMSE2 = np.sqrt(rmse2 / len(test_loader))

end = time.time()
print('Test completed. Program processed ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600, 'h')
print(
    f'Test metrics - 2D: F1 Score -> {mean_f1 * 100} %; mIoU -> {mIoU * 100} %; 3D: MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m')
stats = dict(epoch='Test', MeanAbsoluteError=mean_mae, RMSE=RMSE1, cRMSE=RMSE2, F1Score=mean_f1 * 100, mIoU=mIoU * 100)
print(json.dumps(stats), file=stats_file)
