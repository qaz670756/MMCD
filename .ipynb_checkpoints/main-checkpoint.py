import lightning.pytorch as pl
from torch.utils.data import DataLoader
from models.MTBIT import MTBIT, MTBIT_levir, MTBIT_par_HE, MTBIT_serial_HE
from models.snunet import SNUNet
from models.siamunet_conc import SiamUNet_conc
from models.siamunet_diff import SiamUNet_diff
from models.siamunet_ef import SiamUNet_EF
from models.stanet import STANet
from models.p2v import P2VNet
from models.ChangeFormer import ChangeFormerV6
from models.ifn import DSIFN
from models.ifn_deep3d import DSIFN_DEEP3D
from models.hmcdNet import MixVisionTransformer as hmcdNet
from models.LiteMono import LiteMono

import torch
import numpy as np
import os
import yaml
from argparse import ArgumentParser


from losses import choose_criterion3d, choose_criterion2d
from lightning.pytorch.callbacks import TQDMProgressBar
from LitProgressBar import LitProgressBar
import torch.backends.cudnn as cudnn
import random


def get_args():
    parser = ArgumentParser(description="Hyperparameters", add_help=True)
    parser.add_argument('-c', '--config-name', type=str, help='YAML Config name', 
                        dest='CONFIG', default='MTBIT')
    parser.add_argument('--ckpt_version', type=str, help='checkpoint version', 
                        dest='ckpt_version', default='0')
    parser.add_argument('-nw', '--num-workers', type=int, help='Number of workers', 
                        dest='num_workers', default=0)
    parser.add_argument('-v', '--verbose', type=bool, help='Verbose validation metrics', 
                        dest='verbose', default=False)
    parser.add_argument('--eval', help='train or evalution', 
                        dest='eval', action='store_true')
    parser.add_argument('--save_img', help='save img or not', 
                        dest='save_img', action='store_true')
    parser.add_argument('-d', '--device', type=int, help='device ID', 
                        dest='device', default=0)
    return parser.parse_args()

def parse_args():
    args = get_args()
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
    return args, exp_config, out_file

def define_dataset(exp_config, args):

    mean = exp_config['data']['mean']
    std = exp_config['data']['std']
    if exp_config['data']['augmentations']:
        train_transform = get_training_augmentations(m=mean, s=std)
    else:
        train_transform = get_validation_augmentations(m=mean, s=std)
    valid_transform = get_validation_augmentations(m=mean, s=std)

    x_train_dir = exp_config['data']['train']['path']
    x_valid_dir = exp_config['data']['val']['path']
    x_test_dir = exp_config['data']['test']['path']

    valid_dataset = Dataset(x_valid_dir, exp_config['data']['sets'],
                            augmentation=valid_transform)
    test_dataset = Dataset(x_test_dir, exp_config['data']['sets'],
                           augmentation=valid_transform, return_crs=True)
    train_dataset = Dataset(x_train_dir, exp_config['data']['sets'],
                        augmentation=train_transform)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=args.num_workers, 
                              batch_size=exp_config['data']['batch_size'])
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, 
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             num_workers=args.num_workers)
    
    return train_loader, valid_loader, test_loader

def define_loss(exp_config):
    name_3dloss = exp_config['model']['3d_loss']
    criterion3d = choose_criterion3d(name=name_3dloss)

    weights2d = exp_config['model']['2d_loss_weights']
    class_weights2d = torch.FloatTensor(weights2d)#.cuda()
    name_2dloss = exp_config['model']['2d_loss']
    criterion2d = choose_criterion2d(name_2dloss, class_weights2d)
    
    return criterion2d, criterion3d



def print_model_size(model):

    summary = pl.utilities.model_summary.ModelSummary(model,max_depth=-1)  
    print(summary) 

    from thop import profile
    input_tensor1 = torch.randn((1, 3, 512, 512))
    input_tensor2 = torch.randn((1, 3, 512, 512))
    inputs = (input_tensor1, input_tensor2)

    # Calculate the GFLOPs of the model
    gflops, _ = profile(model.model, inputs=inputs)
    
    print(f"Model GFLOPs: {gflops/1e9:.2f}")
    import pdb;pdb.set_trace()

def main_train(args, exp_config, out_file):

    
    
    criterion2d, criterion3d = define_loss(exp_config)
    train_loader, valid_loader, test_loader = define_dataset(exp_config, args)
    if 'mtbit' in exp_config['model']['model']:
        from cp import pretrain_strategy
        pretrain = exp_config['model']['pretraining_strategy']
        arch = exp_config['model']['feature_extractor_arch']
        CHECKPOINTS = exp_config['model']['checkpoints_path']
        encoder, pretrained, _ = pretrain_strategy(pretrain, CHECKPOINTS, arch)
        if exp_config['model']['model']=='mtbit':
            model = MTBIT(input_nc=3, output_nc=2, 
                               token_len=4, resnet_stages_num=4, 
                               if_upsample_2x=True, 
                               enc_depth=1, dec_depth=8, 
                               decoder_dim_head=16)
        elif exp_config['model']['model']=='mtbit_par_he':
            model = MTBIT_par_HE(input_nc=3, output_nc=2, 
                                 token_len=4, resnet_stages_num=4, 
                                 if_upsample_2x=True,
                                 enc_depth=1, dec_depth=8, decoder_dim_head=16)
            
        elif exp_config['model']['model']=='mtbit_serial_he':
            model = MTBIT_serial_HE(input_nc=3, output_nc=2, 
                                  token_len=4, resnet_stages_num=4, 
                                  if_upsample_2x=True,
                                  enc_depth=1, dec_depth=8, decoder_dim_head=16)
        elif exp_config['model']['model']=='mtbit_levir':
            model = MTBIT_levir(input_nc=3, output_nc=2, 
                               token_len=4, resnet_stages_num=4, 
                               if_upsample_2x=True, 
                               enc_depth=1, dec_depth=8, 
                               decoder_dim_head=16)
    elif exp_config['model']['model']=='snunet':
        model = SNUNet()
    elif exp_config['model']['model']=='siamunet_conc':
        model = SiamUNet_conc()
    elif exp_config['model']['model']=='siamunet_conc_levir':
        from models.siamunet_conc_levir import SiamUNet_conc_levir
        model = SiamUNet_conc_levir()
    elif exp_config['model']['model']=='siamunet_diff':
        model = SiamUNet_diff()
    elif exp_config['model']['model']=='siamunet_ef':
        model = SiamUNet_EF()
    elif exp_config['model']['model']=='stanet':
        model = STANet()
    elif exp_config['model']['model']=='p2vnet':
        model = P2VNet()
    elif exp_config['model']['model']=='changeformer':
        model = ChangeFormerV6(embed_dim=exp_config['model']['emb'])
    elif exp_config['model']['model']=='ifn':
        model = DSIFN()
    elif exp_config['model']['model']=='ifn_deep3d':
        model = DSIFN_DEEP3D()
    elif exp_config['model']['model']=='hmcdnet':
        model = hmcdNet()
    elif exp_config['model']['model']=='litemono':
        model = LiteMono()
        
    pl_model = pl_trainer(model=model, exp_config=exp_config,
                       criterion2d=criterion2d, criterion3d=criterion3d, aux=exp_config['model']['model']=='p2vnet',save_img=args.save_img)
    #print_model_size(pl_model)


    # saves checkpoints to 'some/path/' at every epoch end

    if args.eval!=True:
        trainer = pl.Trainer(default_root_dir=out_file, limit_train_batches=100,
                     max_epochs=exp_config['optim']['num_epochs'], devices=[args.device],# precision=32
                     )
        
        trainer.fit(model=pl_model, train_dataloaders=train_loader#, val_dataloaders=valid_loader,
                   )
    else:
        
        trainer = pl.Trainer(default_root_dir=out_file, limit_train_batches=100,
                     max_epochs=exp_config['optim']['num_epochs'], devices=[args.device], num_nodes=1,logger=None
                     )
        ckpt_path=os.path.join('results',args.CONFIG,'lightning_logs','version_'+args.ckpt_version,'checkpoints')+'/'
        ckpt_path += os.listdir(ckpt_path)[-1]
        
        trainer.test(model=pl_model, dataloaders=test_loader,
                     #ckpt_path='results/CD_serial_he/lightning_logs/version_11/checkpoints/epoch=299-step=30000.ckpt',
                     ckpt_path=ckpt_path
                     )


if __name__ == '__main__':
    
    manual_seed = 18
    
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    
    from dataloader import Dataset
    args, exp_config, out_file = parse_args()
    if len(exp_config['data']['sets'])==5:
        from augmentations_he import get_training_augmentations,get_validation_augmentations
        from pl_model_he import pl_trainer
    elif len(exp_config['data']['sets'])==4:
        from augmentations import get_training_augmentations,get_validation_augmentations
        from pl_model import pl_trainer
    # for levir
    elif len(exp_config['data']['sets'])==3:
        from pl_model_levir import pl_trainer
        from augmentations_levir import get_training_augmentations,get_validation_augmentations
        if 'levir' in exp_config['model']['model']:
            from dataloader_levir import Dataset
        
            
    
    main_train(args, exp_config, out_file)
    
    