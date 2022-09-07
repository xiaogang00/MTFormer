#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import AverageMeter, ProgressMeter, get_output
import torch
import random

def get_loss_meters(p):
    """ Return dictionary with loss meters to monitor training """
    tasks = p.TASKS.NAMES
    losses = {task: AverageMeter('Loss %s' % (task), ':.4e') for task in tasks}
    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses


def train_vanilla(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}

        # Measure loss and performance
        if p['loss_kwargs']['loss_scheme'] == 'baseline_uncertainty':
            output, log_var_list = model(images)
            loss_dict = criterion(output, targets, log_var_list)
        else:
            output = model(images)
            loss_dict = criterion(output, targets)
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results


def train_vanilla2(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    losses['con'] = AverageMeter('Loss con', ':.4e')
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
                             [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        index = batch['index'].cuda(non_blocking=True)

        # Measure loss and performance
        if p['loss_kwargs']['loss_scheme'] == 'baseline_uncertainty':
            output, log_var_list, loss_c = model(images, index=index, inference=False)
            loss_dict = criterion(output, targets, log_var_list)
            loss_c = torch.mean(loss_c) * 0.01
            loss_dict['con'] = loss_c
            loss_dict['total'] += loss_c
        else:
            output = model(images)
            loss_dict = criterion(output, targets)
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES},
                                 {t: targets[t] for t in p.TASKS.NAMES})

        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose=True)

    return eval_results
