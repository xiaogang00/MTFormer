#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x



from models.NCEAverage import NCEAverage
from models.NCECriterion import NCECriterion

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



class MultiTaskModel_uncertainty(nn.Module):
    def __init__(self, backbone: nn.Module, decoders: dict, tasks: list, input_backbone_channels):
        super(MultiTaskModel_uncertainty, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

        num_task = len(tasks)
        self.log_var_list = nn.Parameter(torch.zeros((num_task,), requires_grad=True))

        from models.swim_transformer2 import SwinTransformer as SwinTransformer2
        from models.swim_transformer2_CA_pascal import SwinTransformer as SwinTransformer3

        backbone_channels = input_backbone_channels
        backbone_channels_reduce = 256
        self.backbone_channels_reduce = backbone_channels_reduce
        in_chans = backbone_channels_reduce
        embed_dim = backbone_channels_reduce

        self.decoder = nn.Conv2d(backbone_channels, backbone_channels_reduce, 1)

        self.transformer1 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                             in_chans=in_chans, embed_dim=embed_dim, drop_path_rate=0.0,
                                             out_indices=(0,))

        self.transformer4_1 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                               in_chans=in_chans, embed_dim=embed_dim, drop_path_rate=0.0,
                                               out_indices=(0,))
        self.transformer4_2 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                               in_chans=in_chans, embed_dim=embed_dim, drop_path_rate=0.0,
                                               out_indices=(0,))
        self.transformer4_3 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                               in_chans=in_chans, embed_dim=embed_dim, drop_path_rate=0.0,
                                               out_indices=(0,))

        self.transformer5_1 = SwinTransformer3(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                               in_chans=in_chans * 3, embed_dim=embed_dim * 3, drop_path_rate=0.0,
                                               out_indices=(0,))
        self.transformer5_2 = SwinTransformer3(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                               in_chans=in_chans * 3, embed_dim=embed_dim * 3, drop_path_rate=0.0,
                                               out_indices=(0,))
        self.transformer5_3 = SwinTransformer3(pretrain_img_size=224, window_size=7, depths=(2,), num_heads=(4,),
                                               in_chans=in_chans * 3, embed_dim=embed_dim * 3, drop_path_rate=0.0,
                                               out_indices=(0,))

        key_list = []
        for name in decoders.keys():
            key_list.append(name)
        keys1 = key_list[0]
        keys2 = key_list[1]
        keys3 = key_list[2]
        self.output_channel1 = decoders[keys1]
        self.output_channel2 = decoders[keys2]
        self.output_channel3 = decoders[keys3]
        self.keys1 = keys1
        self.keys2 = keys2
        self.keys3 = keys3

        if 'semseg' in keys1 or 'human_parts' in keys1:
            self.project1 = nn.Sequential(
                nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(backbone_channels_reduce),
                nn.ReLU(True),
                nn.Conv2d(backbone_channels_reduce, self.output_channel1, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        elif 'normals' in keys1 or 'sal' in keys1:
            self.project1 = nn.Sequential(
                nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(backbone_channels_reduce),
                nn.ReLU(True),
                nn.Conv2d(backbone_channels_reduce, self.output_channel1, kernel_size=1),
            )

        if 'semseg' in keys2 or 'human_parts' in keys2:
            self.project2 = nn.Sequential(
                nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(backbone_channels_reduce),
                nn.ReLU(True),
                nn.Conv2d(backbone_channels_reduce, self.output_channel2, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        elif 'normals' in keys2 or 'sal' in keys2:
            self.project2 = nn.Sequential(
                nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(backbone_channels_reduce),
                nn.ReLU(True),
                nn.Conv2d(backbone_channels_reduce, self.output_channel2, kernel_size=1),
            )

        if 'semseg' in keys3 or 'human_parts' in keys3:
            self.project3 = nn.Sequential(
                nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(backbone_channels_reduce),
                nn.ReLU(True),
                nn.Conv2d(backbone_channels_reduce, self.output_channel3, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        elif 'normals' in keys3 or 'sal' in keys3:
            self.project3 = nn.Sequential(
                nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(backbone_channels_reduce),
                nn.ReLU(True),
                nn.Conv2d(backbone_channels_reduce, self.output_channel3, kernel_size=1),
            )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mapping1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 128))
        self.mapping2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 128))
        self.mapping3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 128))
        self.l2_norm = Normalize(2)

        n_data = 4998

        ### 16384
        self.contrast12 = NCEAverage(128, n_data, 2000, 0.07, 0.5).cuda()
        self.criterion1_1 = NCECriterion(n_data).cuda()
        self.criterion1_2 = NCECriterion(n_data).cuda()

        self.contrast13 = NCEAverage(128, n_data, 2000, 0.07, 0.5).cuda()
        self.criterion2_1 = NCECriterion(n_data).cuda()
        self.criterion2_2 = NCECriterion(n_data).cuda()

        self.contrast23 = NCEAverage(128, n_data, 2000, 0.07, 0.5).cuda()
        self.criterion3_1 = NCECriterion(n_data).cuda()
        self.criterion3_2 = NCECriterion(n_data).cuda()

    def forward(self, x, index=0, inference=True):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        shared_representation = self.decoder(shared_representation)
        shared_representation = self.transformer1.forward2(shared_representation)

        feature_T_task1 = self.transformer4_1.forward2(shared_representation)
        feature_T_task2 = self.transformer4_2.forward2(shared_representation)
        feature_T_task3 = self.transformer4_3.forward2(shared_representation)

        feature_T1 = torch.cat([feature_T_task1, feature_T_task2, feature_T_task3], dim=1)
        feature_T1 = self.transformer5_1.forward2(feature_T1)
        feature_T_task1_new = feature_T1[:, 0:self.backbone_channels_reduce, :, :]

        feature_T2 = torch.cat([feature_T_task2, feature_T_task1, feature_T_task3], dim=1)
        feature_T2 = self.transformer5_2.forward2(feature_T2)
        feature_T_task2_new = feature_T2[:, 0:self.backbone_channels_reduce, :, :]

        feature_T3 = torch.cat([feature_T_task3, feature_T_task1, feature_T_task2], dim=1)
        feature_T3 = self.transformer5_3.forward2(feature_T3)
        feature_T_task3_new = feature_T3[:, 0:self.backbone_channels_reduce, :, :]

        ################
        if not (inference):
            feature_1_con = self.pool(feature_T_task1_new).squeeze(dim=3).squeeze(dim=2)
            feature_2_con = self.pool(feature_T_task2_new).squeeze(dim=3).squeeze(dim=2)
            feature_3_con = self.pool(feature_T_task3_new).squeeze(dim=3).squeeze(dim=2)

            feature_1_con = self.mapping1(feature_1_con)
            feature_2_con = self.mapping2(feature_2_con)
            feature_3_con = self.mapping3(feature_3_con)

            feature_1_con = self.l2_norm(feature_1_con)
            feature_2_con = self.l2_norm(feature_2_con)
            feature_3_con = self.l2_norm(feature_3_con)

            out1_1, out1_2 = self.contrast12(feature_1_con, feature_2_con, index)
            loss1_1 = self.criterion1_1(out1_1)
            loss1_2 = self.criterion1_2(out1_2)
            out2_1, out2_2 = self.contrast13(feature_1_con, feature_3_con, index)
            loss2_1 = self.criterion2_1(out2_1)
            loss2_2 = self.criterion2_2(out2_2)
            out3_1, out3_2 = self.contrast23(feature_2_con, feature_3_con, index)
            loss3_1 = self.criterion3_1(out3_1)
            loss3_2 = self.criterion3_2(out3_2)
            loss_con = loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3_1 + loss3_2
        ################

        output_task1 = self.project1(feature_T_task1_new)
        output_task2 = self.project2(feature_T_task2_new)
        output_task3 = self.project3(feature_T_task3_new)

        output = {}
        output[self.keys1] = F.interpolate(output_task1, out_size, mode='bilinear')
        output[self.keys2] = F.interpolate(output_task2, out_size, mode='bilinear')
        output[self.keys3] = F.interpolate(output_task3, out_size, mode='bilinear')

        if inference:
            return output, self.log_var_list
        else:
            return output, self.log_var_list, loss_con
