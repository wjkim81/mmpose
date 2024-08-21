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

def euclidean_distance(pt1, pt2):
    dist = np.linalg.norm(np.array(pt1, dtype=np.float32) - np.array(pt2, dtype=np.float32))

    return dist
    
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

    pred_color = (255, 0, 0)
    gt_color = (0, 255, 0)
    guidewire_color = (0, 0, 255)

    config_path, pth_path = mm_select_work_dir(args.work_dir, args.checkpoint)
    configname = osp.splitext(osp.basename(config_path))[0]
    cfg = Config.fromfile(config_path)
    # print(cfg.test_dataloader)


    device = 'cuda'

    
    # # init model and load checkpoint
    print(f'Initializing model with {config_path} and {pth_path}')
    inferencer = MMPoseInferencer(pose2d=config_path, pose2d_weights=pth_path, device=device)


    modes = ['train', 'test']

    train_dir = osp.join(cfg.train_dataloader.dataset.data_root, cfg.train_dataloader.dataset.data_prefix.img)
    test_dir = osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.data_prefix.img)
    img_dir_dict = {
        'train': train_dir,
        'test': test_dir
    }
    
    annot_path = osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.ann_file)
    annot_path = annot_path.replace('round', 'flat')
    annot_flat_path_dict = {
        'train': osp.join(cfg.train_dataloader.dataset.data_root, cfg.train_dataloader.dataset.ann_file).replace('round', 'flat'),
        'test': osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.ann_file).replace('round', 'flat')
    }
    annot_round_path_dict = {
        'train': osp.join(cfg.train_dataloader.dataset.data_root, cfg.train_dataloader.dataset.ann_file).replace('flat', 'round'),
        'test': osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.ann_file).replace('flat', 'round')
    }

    bigsb_flat_annot_path_dict = {
        'train': osp.join(cfg.train_dataloader.dataset.data_root, 'bigsb-rnd-' + cfg.train_dataloader.dataset.ann_file).replace('round', 'flat'),
        'test': osp.join(cfg.test_dataloader.dataset.data_root, 'bigsb-rnd-' + cfg.test_dataloader.dataset.ann_file).replace('round', 'flat')
    }
    bigsb_round_annot_path_dict = {
        'train': osp.join(cfg.train_dataloader.dataset.data_root, 'bigsb-rnd-' + cfg.train_dataloader.dataset.ann_file).replace('flat', 'round'),
        'test': osp.join(cfg.test_dataloader.dataset.data_root, 'bigsb-rnd-' + cfg.test_dataloader.dataset.ann_file).replace('flat', 'round')
    }

    flat2round = False
    if 'flat' in configname:
        flat2round = True

    save_dir = osp.join(rf'../../data/mmpose/results-onlybigsb/{cfg.test_dataloader.dataset.type}', f'{configname}-kp_range{args.keypoint_range}-gw_thr{args.guidewire_threshold}')

    for mode in modes:
        img_dir = img_dir_dict[mode]
        # save_dir = save_dir_dict[m]
        annot_flat_path = annot_flat_path_dict[mode]
        annot_round_path = annot_round_path_dict[mode]
        dst_flat_annot_path = bigsb_flat_annot_path_dict[mode]
        dst_round_annot_path = bigsb_round_annot_path_dict[mode]

        with open(annot_flat_path, 'r') as json_file:
            annot_flat_data = json.load(json_file)
        with open(annot_round_path, 'r') as json_file:
            annot_round_data = json.load(json_file)

        image_flat_list = annot_flat_data['images']
        annot_flat_list = annot_flat_data['annotations']
        categories_flat_list = annot_flat_data['annotations']

        image_round_list = annot_round_data['images']
        annot_round_list = annot_round_data['annotations']
        categories_round_list = annot_round_data['annotations']

        bigsb_flat_annot_dict = {
            'images': image_flat_list,
            'annotations': [],
            'categories': categories_flat_list
        }
        bigsb_round_annot_dict = {
            'images': image_round_list,
            'annotations': [],
            'categories': categories_round_list
        }
        annot_flat_dict = {}
        annot_round_dict = {}
        for ann_flat in annot_flat_list:
            img_id = ann_flat['image_id']

            if not img_id in annot_flat_dict:
                annot_flat_dict[img_id] = [ann_flat]
            else:
                annot_flat_dict[img_id].append(ann_flat)
        
        for ann_round in annot_round_list:
            img_id = ann_round['image_id']

            if not img_id in annot_round_dict:
                annot_round_dict[img_id] = [ann_round]
            else:
                annot_round_dict[img_id].append(ann_round)

        # Group images by case in dictionary
        cases = {}
        for img_data in image_round_list:
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
            print('-----------------------')
            print('case', case)
            # n_case = len(case_images)
            img_volume = []

            keypoints_pred = {}
            keypoints_gt = {}

            included_index = []
            keypoints_flat_annot ={}
            keypoints_round_annot ={}
            
            for i, img_data in enumerate(case_images):
                file_name = img_data['file_name']
                img_id = img_data['id']

                img_path = osp.join(img_dir, file_name)

                if flat2round:
                    round_path = img_path.replace('flat', 'round')
                else:
                    round_path = img_path

                round_img = cv2.imread(round_path, cv2.IMREAD_UNCHANGED)

                # Keypoints ground truth in flat view
                gt_x_range = []
                if img_id in annot_round_dict:
                    annots_flat = annot_flat_dict[img_id]
                    annots_round = annot_round_dict[img_id]

                    for ann_flat, ann_round in zip(annots_flat, annots_round):
                        include_keypoint = True

                        keypoint_gt = ann_round['keypoints']
                        keypoints = get_keypoints(keypoint_gt)

                        kp1, kp2 = keypoints
                        
                        if kp1[0] > kp2[0]:
                            tmp = kp2
                            kp2 = kp1
                            kp1 = tmp
                            keypoints = (kp1, kp2)

                        include_keypoint = True

                        gt_x_dist = euclidean_distance(kp1, kp2)

                        # if gt_x_dist < args.keypoint_range:
                        #     print(f'{i} small gt keypoint - length: {gt_x_dist}')
                        #     include_keypoint = False

                        if include_keypoint:
                            keypoints_gt[i] = {
                                'keypoints': keypoints,
                                'keypoints_dist': gt_x_dist
                            }
                            included_index.append(i)
                            keypoints_flat_annot[i] = ann_flat
                            keypoints_round_annot[i] = ann_round

            print('included_index:', included_index)
            included_index = list(set(included_index))
            continuous_list = group_continuous(included_index)
            # continuous_set = set(continuous_list)
            print('continuous_list:', continuous_list)
            # print('continuous_set:', continuous_set)
            # print('keypoints_gt:', keypoints_gt)

            for cl in continuous_list:
                include_keypoint = False
                if len(cl) > 3:
                    for idx in cl:
                        # keypoints = keypoints_gt[idx]
                        keypoints_dist = keypoints_gt[idx]['keypoints_dist']
                        if keypoints_dist > args.keypoint_range:
                            include_keypoint = True
                    if include_keypoint:
                        for idx in cl:
                            bigsb_flat_annot_dict['annotations'].append(keypoints_flat_annot[idx])
                            bigsb_round_annot_dict['annotations'].append(keypoints_round_annot[idx])
                else:
                    for idx in cl:
                        print(idx)
                        if idx in keypoints_gt:
                            keypoints_gt.pop(idx)

        
        print(f'Saving big side branch extracted annotation to {dst_flat_annot_path}')
        print(f'Saving big side branch extracted annotation to {dst_round_annot_path}')
        with open(dst_flat_annot_path, 'w') as annot_json:
            json.dump(bigsb_flat_annot_dict, annot_json, indent=4)
        with open(dst_round_annot_path, 'w') as annot_json:
            json.dump(bigsb_round_annot_dict, annot_json, indent=4)
