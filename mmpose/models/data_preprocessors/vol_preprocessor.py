# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import ImgDataPreprocessor

from mmpose.registry import MODELS


@MODELS.register_module()
class VolDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks."""
