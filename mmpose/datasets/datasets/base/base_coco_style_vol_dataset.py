# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from copy import deepcopy
from itertools import filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import exists, get_local_path, load
from mmengine.utils import is_list_of
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_xywh2xyxy
from ..utils import parse_pose_metainfo

import pickle

@DATASETS.register_module()
class BaseCocoStyleVolDataset(BaseDataset):
    """Base class for COCO-style datasets.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 n_frames = 5):

        if data_mode not in {'topdown', 'bottomup'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid data_mode: '
                f'{data_mode}. Should be "topdown" or "bottomup".')
        self.data_mode = data_mode

        if bbox_file:
            if self.data_mode != 'topdown':
                raise ValueError(
                    f'{self.__class__.__name__} is set to {self.data_mode}: '
                    'mode, while "bbox_file" is only '
                    'supported in topdown mode.')

            if not test_mode:
                raise ValueError(
                    f'{self.__class__.__name__} has `test_mode==False` '
                    'while "bbox_file" is only '
                    'supported when `test_mode==True`.')
        self.bbox_file = bbox_file
        self.n_frames = n_frames

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        # parse pose metainfo if it has been assigned
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    @force_full_init
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        :class:`BaseCocoStyleDataset` overrides this method from
        :class:`mmengine.dataset.BaseDataset` to add the metainfo into
        the ``data_info`` before it is passed to the pipeline.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)

        return self.pipeline(data_info)

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index.

        Args:
            idx (int): Index of data info.

        Returns:
            dict: Data info.
        """
        data_info = super().get_data_info(idx)

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices', 'skeleton_links'
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                'exists in the `data_info`.')

            data_info[key] = deepcopy(self._metainfo[key])

        return data_info

    def load_data_list(self) -> List[dict]:
        """Load data list from COCO annotation file or person detection result
        file."""

        if self.bbox_file:
            data_list = self._load_detection_results()
        else:
            instance_list, image_list = self._load_annotations()

            if self.data_mode == 'topdown':
                data_list = self._get_topdown_data_infos(instance_list)
            else:
                data_list = self._get_bottomup_data_infos(
                    instance_list, image_list)

        case_list = []
        case_dict_list = {}
        for img_data in data_list:
            case = img_data['case']
            # print('case:', case)

            if case[0] not in case_dict_list:
                case_dict_list[case[0]] = [img_data]
            else:
                case_dict_list[case[0]].append(img_data)

        for case, img_list in case_dict_list.items():
            case_list.append(img_list)


        # which one?????
        # return data_list
        return case_list
        # return case_dict_list

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""

        assert exists(self.ann_file), 'Annotation file does not exist'

        with get_local_path(self.ann_file) as local_path:
            # print('local_path:', osp.abspath(local_path))
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        self._metainfo['CLASSES'] = self.coco.loadCats(self.coco.getCatIds())

        instance_list = []
        image_list = []

        for img_id in self.coco.getImgIds():
            img = self.coco.loadImgs(img_id)[0]
            img.update({
                'img_id':
                img_id,
                'img_path':
                osp.join(self.data_prefix['img'], img['file_name']),
            })
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            for ann in self.coco.loadAnns(ann_ids):

                instance_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=img))

                # skip invalid instance annotation.
                if not instance_info:
                    continue

                instance_list.append(instance_info)
        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': ann['category_id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),

            'case': ann['case']
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info

    @staticmethod
    def _is_valid_instance(data_info: Dict) -> bool:
        """Check a data info is an instance with valid bbox and keypoint
        annotations."""
        # crowd annotation
        if 'iscrowd' in data_info and data_info['iscrowd']:
            return False
        # invalid keypoints
        if 'num_keypoints' in data_info and data_info['num_keypoints'] == 0:
            return False
        # invalid bbox
        if 'bbox' in data_info:
            bbox = data_info['bbox'][0]
            w, h = bbox[2:4] - bbox[:2]
            if w <= 0 or h <= 0:
                return False
        # invalid keypoints
        if 'keypoints' in data_info:
            if np.max(data_info['keypoints']) <= 0:
                return False
        return True

    def _get_topdown_data_infos(self, instance_list: List[Dict]) -> List[Dict]:
        """Organize the data list in top-down mode."""
        # sanitize data samples
        data_list_tp = list(filter(self._is_valid_instance, instance_list))

        return data_list_tp

    def _get_bottomup_data_infos(self, instance_list: List[Dict],
                                 image_list: List[Dict]) -> List[Dict]:
        """Organize the data list in bottom-up mode."""

        # bottom-up data list
        data_list_bu = []

        used_img_ids = set()

        # group instances by img_id
        for img_id, data_infos in groupby(instance_list,
                                          lambda x: x['img_id']):
            used_img_ids.add(img_id)
            data_infos = list(data_infos)

            # image data
            img_path = data_infos[0]['img_path']
            data_info_bu = {
                'img_id': img_id,
                'img_path': img_path,
            }

            for key in data_infos[0].keys():
                if key not in data_info_bu:
                    seq = [d[key] for d in data_infos]
                    if isinstance(seq[0], np.ndarray):
                        seq = np.concatenate(seq, axis=0)
                    data_info_bu[key] = seq

            # The segmentation annotation of invalid objects will be used
            # to generate valid region mask in the pipeline.
            invalid_segs = []
            for data_info_invalid in filterfalse(self._is_valid_instance,
                                                 data_infos):
                if 'segmentation' in data_info_invalid:
                    invalid_segs.append(data_info_invalid['segmentation'])
            data_info_bu['invalid_segs'] = invalid_segs

            data_list_bu.append(data_info_bu)

        # add images without instance for evaluation
        if self.test_mode:
            for img_info in image_list:
                if img_info['img_id'] not in used_img_ids:
                    data_info_bu = {
                        'img_id': img_info['img_id'],
                        'img_path': img_info['img_path'],
                        'id': list(),
                        'raw_ann_info': None,
                    }
                    data_list_bu.append(data_info_bu)

        return data_list_bu

    def _load_detection_results(self) -> List[dict]:
        """Load data from detection results with dummy keypoint annotations."""

        assert exists(self.ann_file), 'Annotation file does not exist'
        assert exists(self.bbox_file), 'Bbox file does not exist'
        # load detection results
        det_results = load(self.bbox_file)
        assert is_list_of(det_results, dict)

        # load coco annotations to build image id-to-name index
        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        self._metainfo['CLASSES'] = self.coco.loadCats(self.coco.getCatIds())

        num_keypoints = self.metainfo['num_keypoints']
        data_list = []
        id_ = 0
        for det in det_results:
            # remove non-human instances
            if det['category_id'] != 1:
                continue

            img = self.coco.loadImgs(det['image_id'])[0]

            img_path = osp.join(self.data_prefix['img'], img['file_name'])
            bbox_xywh = np.array(
                det['bbox'][:4], dtype=np.float32).reshape(1, 4)
            bbox = bbox_xywh2xyxy(bbox_xywh)
            bbox_score = np.array(det['score'], dtype=np.float32).reshape(1)

            # use dummy keypoint location and visibility
            keypoints = np.zeros((1, num_keypoints, 2), dtype=np.float32)
            keypoints_visible = np.ones((1, num_keypoints), dtype=np.float32)

            data_list.append({
                'img_id': det['image_id'],
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'bbox_score': bbox_score,
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'id': id_,
            })

            id_ += 1

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg. Defaults return full
        ``data_list``.

        If 'bbox_score_thr` in filter_cfg, the annotation with bbox_score below
        the threshold `bbox_score_thr` will be filtered out.
        """

        data_list = self.data_list

        if self.filter_cfg is None:
            return data_list

        # filter out annotations with a bbox_score below the threshold
        if 'bbox_score_thr' in self.filter_cfg:

            if self.data_mode != 'topdown':
                raise ValueError(
                    f'{self.__class__.__name__} is set to {self.data_mode} '
                    'mode, while "bbox_score_thr" is only supported in '
                    'topdown mode.')

            thr = self.filter_cfg['bbox_score_thr']
            data_list = list(
                filterfalse(lambda ann: ann['bbox_score'] < thr, data_list))

        return data_list

    # def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """Serialize ``self.data_list`` to save memory when launching multiple
    #     workers in data loading. This function will be called in ``full_init``.

    #     Hold memory using serialized objects, and data loader workers can use
    #     shared RAM from master process instead of making a copy.

    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
    #         address.
    #     """

    #     def _serialize(data):
    #         buffer = pickle.dumps(data, protocol=4)
    #         return np.frombuffer(buffer, dtype=np.uint8)

    #     # Serialize data information list avoid making multiple copies of
    #     # `self.data_list` when iterate `import torch.utils.data.dataloader`
    #     # with multiple workers.
    #     data_list = [_serialize(x) for x in self.data_list]
    #     address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
    #     data_address: np.ndarray = np.cumsum(address_list)
    #     # TODO Check if np.concatenate is necessary
    #     data_bytes = np.concatenate(data_list)
    #     # Empty cache for preventing making multiple copies of
    #     # `self.data_info` when loading data multi-processes.
    #     self.data_list.clear()
    #     gc.collect()
    #     return data_bytes, data_address
    

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """

        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info