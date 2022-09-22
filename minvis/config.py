# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_minvis_config(cfg):
    cfg.INPUT.SAMPLING_FRAME_RATIO = 1.0
    cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE = False

    # BatchFormer
    cfg.MODEL.MASK_FORMER.SHARE_BF = 0
    cfg.MODEL.MASK_FORMER.BF = 1
    cfg.MODEL.MASK_FORMER.INSERT_IDX = [0]
    cfg.MODEL.MASK_FORMER.BT_NUM_LAYERS = 1
    cfg.MODEL.MASK_FORMER.EVAL_BF = False
