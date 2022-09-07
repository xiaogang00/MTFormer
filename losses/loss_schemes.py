#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, pred, gt):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        return out


class MultiTaskLoss_uncertainty(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss_uncertainty, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt, log_vars):
        loss_weight_puls = {}
        loss_weight_mul = {}
        count = 0
        for task in self.tasks:
            if 'semseg' in task:
                precision = torch.exp(-log_vars[count])
            elif 'depth' in task:
                precision = torch.exp(-log_vars[count])
            elif 'human_parts' in task:
                precision = torch.exp(-log_vars[count])
            elif 'sal' in task:
                precision = torch.exp(-log_vars[count])
            loss_weight_mul[task] = precision
            loss_weight_puls[task] = log_vars[count]
            count += 1

        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * loss_weight_mul[t] * 10 * out[t] + loss_weight_puls[t] for t in self.tasks])) ### original
        return out

