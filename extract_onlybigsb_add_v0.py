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
    base_dir = r'D:\data\datasets\octseg'
    modes = ['train', 'test']
    
    annot_dict = {
        'flat_train': 'sidebranch_flat_train.json',
        'flat_test': 'sidebranch_flat_test.json',
        'round_train': 'sidebranch_round_train.json',
        'round_test': 'sidebranch_round_test.json',
        'big-flat-train': 'big-sidebranch_flat_train.json',
        'big-flat-test': 'big-sidebranch_flat_test.json',
        'big-round-train': 'big-sidebranch_round_train.json',
        'big-round-test': 'big-sidebranch_round_test.json',
    }

    for key, json_fn in annot_dict.items():
        print(key, json_fn)
        annot_path = osp.join(base_dir, json_fn)
        v0_path = osp.join(base_dir, 'v0-' + json_fn)

        with open(annot_path, 'r') as json_file:
            annot_data = json.load(json_file)
        
        image_list = annot_data['images']
        annot_list = annot_data['annotations']
        categories = annot_data['categories']

        annot_dict = {}
        for ann in annot_list:
            img_id = ann['image_id']

            if not img_id in annot_dict:
                annot_dict[img_id] = [ann]
            else:
                annot_dict[img_id].append(ann)

        v0_annot_dict = {
            'images': image_list,
            # 'annotations': [],
            'categories': categories
        }

        annot_x = {
            "segmentation": [[]],
            "keypoints": [0, 0, 0, 0, 0, 0 ],
            "num_keypoints": 2,
            "iscrowd": 0,
            # "image_id": 22,
            "category_id": 1,
            # "id": 1,
            "area": 262144,
            "bbox": [0, 0, 0, 0]          
        }

        annot_id = 1
        v0_annot_list = []
        for img_dict in image_list:
            img_id = img_dict['id']
            file_name = img_dict['file_name']
            case, fn = file_name.split('/')
            # print('img_id:', img_id)

            if img_id in annot_dict:
                annots = annot_dict[img_id]
                for ann in annots:
                    ann['id'] = annot_id
                    ann['case'] = case
                    # ann['file_name'] = file_name
                    v0_annot_list.append(ann)
                    annot_id += 1
            else:
                annot = annot_x.copy()
                annot['image_id'] = img_id
                annot['id'] = annot_id
                annot['case'] = case
                # annot['file_name'] = file_name
                v0_annot_list.append(annot)
                annot_id += 1

            # print('annot_id:', annot_id)
        
        v0_annot_dict['annotations'] = v0_annot_list
        print(f'Saving big side branch extracted annotation to {v0_path}')
        with open(v0_path, 'w') as annot_json:
            json.dump(v0_annot_dict, annot_json, indent=4)
        