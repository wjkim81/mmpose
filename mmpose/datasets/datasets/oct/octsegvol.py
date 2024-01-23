# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleVolDataset


@DATASETS.register_module(name='OCTSegVolDataset')
class OCTSegVolDataset(BaseCocoStyleVolDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/octseg.py')