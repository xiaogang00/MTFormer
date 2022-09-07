#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import copy
import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import numpy as np

"""
    Model getters 
"""


def load_checkpoint(model, checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    state_dict = checkpoint['model']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
        # reshape absolute position embedding
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            print("Error in loading absolute_pos_embed, pass")
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)
            # interpolate position bias table if needed
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                table_pretrained_resized = F.interpolate(
                    table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
    # load state_dict
    return state_dict


def get_backbone(p):
    """ Return the backbone """

    if p['backbone'] == 'swim_transformer2':
        from models.swim_transformer2 import SwinTransformer

        ### swin-S
        backbone = SwinTransformer(pretrain_img_size=224, window_size=7, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
                                   embed_dim=96, drop_path_rate=0.3)
        pretrain_path = '/mnt/backup2/home/xgxu/Multi-Task-Learning-PyTorch-master/pre_train/swin_small_patch4_window7_224.pth'

        state_dict = load_checkpoint(backbone, pretrain_path)

        unexpected_keys = []
        all_missing_keys = []
        err_msg = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(backbone)
        load = None  # break load->load reference cycle
        # ignore "num_batches_tracked" of BN layers
        missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
        if unexpected_keys:
            err_msg.append('unexpected key in source ' f'state_dict: {", ".join(unexpected_keys)}\n')
        if missing_keys:
            err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')
        print(err_msg)
        del state_dict

        backbone_channels = 768+384+192 ## swin-S
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)

    if p['setup'] == 'multi_task':
        if p['model'] == 'MTFormer_nyud':
            from models.models_nyud_CL import MultiTaskModel_uncertainty
            output_list = {}
            for task in p.TASKS.NAMES:
                output_list[task] = p.TASKS.NUM_OUTPUT[task]
            model = MultiTaskModel_uncertainty(backbone, output_list, p.TASKS.NAMES, backbone_channels)
        elif p['model'] == 'MTFormer_pascal':
            from models.models_pascal_CL import MultiTaskModel_uncertainty
            output_list = {}
            for task in p.TASKS.NAMES:
                output_list[task] = p.TASKS.NUM_OUTPUT[task]
            model = MultiTaskModel_uncertainty(backbone, output_list, p.TASKS.NAMES, backbone_channels)
        else:
            raise NotImplementedError('Unknown model {}'.format(p['model']))

    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))

    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import custom_transforms as tr

    # Training transformations
    if p['train_db_name'] == 'NYUD':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=[0], scales=[1.0, 1.2, 1.5],
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])

    elif p['train_db_name'] == 'PASCALContext':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])
    else:
        raise ValueError('Invalid train db name'.format(p['train_db_name']))


    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(p.TRAIN.SCALE) for x in p.ALL_TASKS.FLAGVALS},
                                         flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])
    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_tr = transforms.Compose(transforms_tr)

    
    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(p.TEST.SCALE) for x in p.TASKS.FLAGVALS},
                                         flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts


def get_train_dataset(p, transforms):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train loader for db: {}'.format(db_name))

    if db_name == 'PASCALContext2':
        from data.pascal_context2 import PASCALContext
        database = PASCALContext(split=['train'], transform=transforms, retname=True,
                                    do_semseg='semseg' in p.ALL_TASKS.NAMES,
                                    do_sal='sal' in p.ALL_TASKS.NAMES,do_human_parts='human_parts' in p.ALL_TASKS.NAMES,
                                    overfit=p['overfit'])

    elif db_name == 'NYUD2':
        from data.nyud2 import NYUD_MT
        database = NYUD_MT(split='train', transform=transforms,
                                    do_semseg='semseg' in p.ALL_TASKS.NAMES,
                                    do_depth='depth' in p.ALL_TASKS.NAMES, overfit=p['overfit'])

    else:
        raise NotImplemented("train_db_name: Choose among PASCALContext and NYUD")

    return database


def get_train_dataloader(p, dataset):
    """ Return the train dataloader """

    trainloader = DataLoader(dataset, batch_size=p['trBatch'], shuffle=True, drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate_mil)
    return trainloader


def get_val_dataset(p, transforms):
    """ Return the validation dataset """

    db_name = p['val_db_name']
    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                    overfit=p['overfit'])
    
    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(split='val', transform=transforms,
                                do_semseg='semseg' in p.TASKS.NAMES,
                                do_depth='depth' in p.TASKS.NAMES, overfit=p['overfit'])
    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_val_dataloader(p, dataset):
    """ Return the validation dataloader """

    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'])
    return testloader


""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import SoftMaxwithLoss
        criterion = SoftMaxwithLoss()

    elif task == 'sal':
        from losses.loss_functions import BalancedCrossEntropyLoss
        criterion = BalancedCrossEntropyLoss(size_average=True)

    elif task == 'depth':
        from losses.loss_functions import DepthLoss
        criterion = DepthLoss(p['depthloss'])

    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'semseg, human_parts, sal, depth')

    return criterion


def get_criterion(p):
    """ Return training criterion for a given setup """

    if p['setup'] == 'multi_task':
        if p['loss_kwargs']['loss_scheme'] == 'baseline_uncertainty': # Fixed weights
            from losses.loss_schemes import MultiTaskLoss_uncertainty
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return MultiTaskLoss_uncertainty(p.TASKS.NAMES, loss_ft, loss_weights)
        
        else:
            raise NotImplementedError('Unknown loss scheme {}'.format(p['loss_kwargs']['loss_scheme']))

    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    print('Optimizer uses a single parameter group - (Default)')
    params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])
    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=p['optimizer_kwargs']['lr'],
                                      betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=p['optimizer_kwargs']['weight_decay'], amsgrad=False)

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    """ Adjust the learning rate """

    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1-(epoch/p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate2(p, optimizer, epoch, epoch_total):
    """ Adjust the learning rate """

    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / epoch_total), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
