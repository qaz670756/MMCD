import torch.optim as optim


def set_optimizer(optim_params, model):
    if optim_params['optim_type'] == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            lr = optim_params['lr'],
            betas = optim_params['beta'],
            weight_decay = optim_params['weight_decay'])
    elif optim_params['optim_type'] == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
            lr = optim_params['lr'], 
            betas = optim_params['beta'],
            weight_decay= optim_params['weight_decay'])
    elif optim_params['optim_type'] == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
            lr = optim_params['lr'],
            momentum = optim_params['momentum'],
            nesterov = optim_params['nesterov'],
            weight_decay = optim_params['weight_decay'])
    else:
        raise Exception('The selected optimization type is not available.')

    return optimizer

def set_scheduler(optim_params, optimizer):
    if optim_params['lr_schedule_type'] == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size = optim_params['step'], 
                                              gamma = optim_params['gamma'],
                                              )
        # scheduler.last_epoch = optim_params['last_epoch']
    elif optim_params['lr_schedule_type'] == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 
                                              optim_params['gamma'])
    elif optim_params['lr_schedule_type'] == 'red_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                              mode = 'min', 
                                              factor = optim_params['gamma'], 
                                              patience = optim_params['lr_patience'],
                                              min_lr = optim_params['end_lr'])
    elif optim_params['lr_schedule_type'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         T_max = 10, 
                                                         eta_min=optim_params['end_lr'])
    else:
        raise Exception('The selected scheduler type is not available.')
    return scheduler