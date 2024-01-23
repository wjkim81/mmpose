import os
import os.path as osp
import json

import torch
import argparse
import cv2
import numpy as np

from operator import itemgetter
from itertools import groupby

from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from octmodules.warping import point_polar_to_cart
from octmodules.mmutils import draw_img_with_mask
from octmodules.mmutils import mm_select_work_dir

def get_keypoints(keypoint_list):
    pt1 = keypoint_list[0:2]
    pt2 = keypoint_list[3:5]
    return (pt1, pt2)

def warp_keypoints_to_cart(keypoints, img_sz):
    cart_keypoints = []
    for kp in keypoints:
        cart_keypoints.append(point_polar_to_cart(kp, img_sz))

    return cart_keypoints

def group_continuous(included_index):
    ret = []
    for k, g in groupby(enumerate(included_index), lambda ix : ix[0]-ix[1]):
        ret.append(list(map(itemgetter(1), g)))

    return ret

# Estimate metrics using only flat images!!
if __name__ == '__main__':
    work_dir = '../../data/mmpose/work_dirs'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', type=str, default=work_dir,
        help='specify working directory of trainined model')
    parser.add_argument('--checkpoint', type=str, default='last', choices=['last', 'best'],
        help='select which chekpoint will be chosen [last|best]')
    parser.add_argument('--threshold', type=float, default=0.3,
        help='the threshold of keypoiont score')
    parser.add_argument('--keypoint_range', type=int, default=30,
        help='the range which include the correct prediction')
    parser.add_argument('--guidewire_threshold', type=float, default=0.5,
        help='threshold of duplicate of guidewires')

    args = parser.parse_args()

    config_path, pth_path = mm_select_work_dir(args.work_dir, args.checkpoint)
    configname = osp.splitext(osp.basename(config_path))[0]
    cfg = Config.fromfile(config_path)
    # print(cfg.test_dataloader)

    modes = ['train_flat', 'test_flat', 'train_round', 'test_round', 'train_bigsb_flat', 'test_bigsb_flat', 'train_bigsb_round', 'test_bigsb_round']
    
    annot_path_dict = {
        'train_flat': osp.join(cfg.train_dataloader.dataset.data_root, cfg.train_dataloader.dataset.ann_file).replace('round', 'flat'),
        'test_flat': osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.ann_file).replace('round', 'flat'),
        'train_round': osp.join(cfg.train_dataloader.dataset.data_root, cfg.train_dataloader.dataset.ann_file).replace('flat', 'round'),
        'test_round': osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.ann_file).replace('flat', 'round'),
        'train_bigsb_flat': osp.join(cfg.train_dataloader.dataset.data_root, 'bigsb-' + cfg.train_dataloader.dataset.ann_file).replace('round', 'flat'),
        'test_bigsb_flat': osp.join(cfg.test_dataloader.dataset.data_root, 'bigsb-' + cfg.test_dataloader.dataset.ann_file).replace('round', 'flat'),
        'train_bigsb_round': osp.join(cfg.train_dataloader.dataset.data_root, cfg.train_dataloader.dataset.ann_file).replace('flat', 'round'),
        'test_bigsb_round': osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.ann_file).replace('flat', 'round')
    }

    vol_annot_path_dict = {
        'train_flat': osp.join(cfg.train_dataloader.dataset.data_root, 'vol-' + cfg.train_dataloader.dataset.ann_file).replace('round', 'flat'),
        'test_flat': osp.join(cfg.test_dataloader.dataset.data_root, 'vol-' + cfg.test_dataloader.dataset.ann_file).replace('round', 'flat'),
        'train_round': osp.join(cfg.train_dataloader.dataset.data_root, 'vol-' + cfg.train_dataloader.dataset.ann_file).replace('flat', 'round'),
        'test_round': osp.join(cfg.test_dataloader.dataset.data_root, 'vol-' + cfg.test_dataloader.dataset.ann_file).replace('flat', 'round'),
        'train_bigsb_flat': osp.join(cfg.train_dataloader.dataset.data_root, 'vol-bigsb-' + cfg.train_dataloader.dataset.ann_file).replace('round', 'flat'),
        'test_bigsb_flat': osp.join(cfg.test_dataloader.dataset.data_root, 'vol-bigsb-' + cfg.test_dataloader.dataset.ann_file).replace('round', 'flat'),
        'train_bigsb_round': osp.join(cfg.train_dataloader.dataset.data_root, 'vol-bigsb-' + cfg.train_dataloader.dataset.ann_file).replace('flat', 'round'),
        'test_bigsb_round': osp.join(cfg.test_dataloader.dataset.data_root, 'vol-bigsb-' + cfg.test_dataloader.dataset.ann_file).replace('flat', 'round')
    }

    for mode in modes:
        annot_path = annot_path_dict[mode]
        vol_annot_path = vol_annot_path_dict[mode]
        print(f'{annot_path} -> {vol_annot_path}')
        
        
        with open(annot_path, 'r') as json_file:
            annot_data = json.load(json_file)

        image_list = annot_data['images']
        annot_list = annot_data['annotations']

        annot_dict = {}
        for ann_flat in annot_list:
            img_id = ann_flat['image_id']

            if not img_id in annot_dict:
                annot_dict[img_id] = [ann_flat]
            else:
                annot_dict[img_id].append(ann_flat)
        
        # Group images by case in dictionary
        cases = {}
        cases_list = []
        for img_data in image_list:
            file_name = img_data['file_name']
            # img_id = img_data['id']

            # img_path = osp.join(test_dir, file_name)
            case, fn = file_name.split('/')
            # print(f'case: {case}, fi: {fi}')

            if case not in cases:
                cases[case] = [img_data]
            else:
                cases[case].append(img_data)

        for case, case_images in cases.items():
            # print('-----------------------')
            cases_list.append(case_images)

        group_annot_dict = {
            'images': cases_list,
            'annotations': annot_list
        }

        print(f'Saving big side branch extracted annotation to {vol_annot_path}')
        with open(vol_annot_path, 'w') as annot_json:
            json.dump(group_annot_dict, annot_json, indent=4)