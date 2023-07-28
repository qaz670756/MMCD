import torch
t = torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def pretrain_strategy(pretrained, cp_path, arch = 'resnet18', n_classes = 2): #resnet34

  if pretrained == 'obow':
    cp = torch.load(cp_path)['network']
    n_classes = len(cp[list(cp.keys())[-1]])
    encoder = models.__dict__[arch](num_classes = n_classes)
    encoder.load_state_dict(cp)
    
#   https://github.com/vturrisi/solo-learn
  elif pretrained == 'ssl_imagenet':
    cp = torch.load(cp_path)
    d = cp['state_dict']
    d_n = {}
    for k in d.keys():
      if 'backbone' in k:
        k_n = k.replace('backbone.', '')
        d_n[k_n] = d[k]
    
    encoder = models.__dict__[arch](num_classes = n_classes)
    encoder.load_state_dict(d_n, strict = False)
      

  elif pretrained == 'imagenet':
    encoder = models.__dict__[arch](weights='ResNet18_Weights.DEFAULT')
    n_classes = list(encoder.children())[-1].out_features

  elif pretrained == 'no':
    encoder = models.__dict__[arch]()
    n_classes = list(encoder.children())[-1].out_features
    pretrained = 'no pretraining'

  else:
    raise Exception('Pretrained strategy not supported')
    
  print('Encoder selected: ' + arch + '\nPretrained with the following strategy: ' + pretrained)

  return encoder, pretrained, n_classes