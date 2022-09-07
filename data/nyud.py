# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import sys
import tarfile
import cv2

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

from utils.mypath import MyPath
from utils.utils import mkdir_if_missing


class NYUD_MT(data.Dataset):

    def __init__(self,
                 root=MyPath.db_root_dir('NYUD_MT'),
                 split='val',
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_semseg=False,
                 do_depth=False,
                 ):

        self.root = root

        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'images')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                # Images
                _image = os.path.join(_image_dir, line + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Semantic Segmentation
                _semseg = os.path.join(self.root, _semseg_gt_dir, line + '.png')
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                # Depth Prediction
                _depth = os.path.join(self.root, _depth_gt_dir, line + '.npy')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                print('RESHAPE SEMSEG')
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_depth:
            _depth = self._load_depth(index)
            if _depth.shape[:2] != _img.shape[:2]:
                print('RESHAPE DEPTH')
                _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_semseg(self, index):
        # Note: We ignore the background class as other related works.
        _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)
        _semseg[_semseg == 0] = 256
        _semseg = _semseg - 1
        return _semseg

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        return _depth

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'

