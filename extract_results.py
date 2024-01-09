import os
import os.path as osp
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary

from mmengine.config import Config
# from mmpose.apis import MMPoseInferencer
from mmpose.apis import init_model
from mmengine.dataset import Compose, pseudo_collate

from octmodules.mmutils import mm_select_work_dir
import os
import os.path as osp
import json
import glob
import argparse
import cv2
import numpy as np

from mmengine.config import Config

from mmpose.apis import inference_topdown, init_model
from mmpose.apis import MMPoseInferencer

from octmodules.mmutils import mm_select_work_dir


def get_keypoints(keypoint_list):
    pt1 = keypoint_list[0:2]
    pt2 = keypoint_list[3:5]
    return (pt1, pt2)

if __name__ == '__main__':
    work_dir = '../../data/mmpose/work_dirs'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', type=str, default=work_dir,
        help='specify working directory of trainined model')
    parser.add_argument('--checkpoint', type=str, default='last', choices=['last', 'best'],
        help='select which chekpoint will be chosen [last|best]')
    parser.add_argument('--threshold', type=float, default=0.3,
        help='the threshold of keypoiont score')
    args = parser.parse_args()

    pred_color = (255, 0, 0)
    gt_color = (0, 255, 0)

    config_path, pth_path = mm_select_work_dir(args.work_dir, args.checkpoint)
    configname = osp.splitext(osp.basename(config_path))[0]
    cfg = Config.fromfile(config_path)
    # print(cfg.test_dataloader)

    # # init model and load checkpoint
    print(f'Initializing model with {config_path} and {pth_path}')
    # model = init_model(config_path, pth_path, device='cuda')
    inferencer = MMPoseInferencer(pose2d=config_path, pose2d_weights=pth_path)

    # img_dir = osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.data_prefix.img)
    img_dirs = {
        'train': osp.join(cfg.train_dataloader.dataset.data_root, cfg.train_dataloader.dataset.data_prefix.img),
        'test': osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.data_prefix.img)
    }
    
    annot_paths = {
        'train': osp.join(cfg.train_dataloader.dataset.data_root, cfg.train_dataloader.dataset.ann_file),
        'test': osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.ann_file)
    }

    
    save_paths = {
        'train': osp.join(rf'../../data/mmpose/results/{cfg.train_dataloader.dataset.type}', configname + '-train.json'),
        'test': osp.join(rf'../../data/mmpose/results/{cfg.test_dataloader.dataset.type}', configname + '-test.json')
    }
    

    for mode in ['train', 'test']:
        with open(annot_paths[mode], 'r') as json_file:
            annot_data = json.load(json_file)

        data_dict = {
            'metainfo': {'classes': ['Non-Sidebranch', 'Sidebranch']}
        }

        data_list = []

        image_list = annot_data['images']
        annot_list = annot_data['annotations']

        annot_dict = {}
        for ann in annot_list:
            img_id = ann['image_id']

            if not img_id in annot_dict:
                annot_dict[img_id] = [ann]
            else:
                annot_dict[img_id].append(ann)
            
        for img_data in image_list:
            file_name = img_data['file_name']
            img_id = img_data['id']

            img_path = osp.join(img_dirs[mode], file_name)

            # data_list
            data = {'img_path': img_path}
          
            # print('**************************')
            
            keypoints_gt = []
            gt_label = 0
            if img_id in annot_dict:
                annots = annot_dict[img_id]

                for ann in annots:
                    keypoint_gt = ann['keypoints']
                    # print(keypoint_gt)
                    keypoints = get_keypoints(keypoint_gt)
                    # print(keypoints)
                    keypoints_gt.append(keypoints)

                if len(keypoints_gt) > 0:
                    gt_label = 1
            
            data['gt_label'] = gt_label
            data['keypoints_gt'] = keypoints_gt

            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
            # case, fn = file_name.split('/')
            # img_fn = file_name.replace('/', '-')

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result_generator = inferencer(img_path)
            result = next(result_generator)
            
            predictions = result['predictions']

            keypoints_pred = []
            pred_label = 0
            for preds in predictions:
            #     print('preds:', preds)
                for p in preds:
            #         print('p', p)
                    kp_pred = p['keypoints']
                    kp_scores = p['keypoint_scores']
                    # print('keypoints:', keypoints_pred)
                    # print('keypoint_scores:', keypoint_scores)

                    keypoint_exist = False
                    for ks in kp_scores:
                        if ks > args.threshold: keypoint_exist = True

                    if keypoint_exist:
                        kp1 = [int(x + 0.5) for x in kp_pred[0]]
                        kp2 = [int(x + 0.5) for x in kp_pred[1]]

                        if kp1[1] > kp2[1]:
                            tmp = kp2
                            kp2 = kp1
                            kp1 = tmp

                        keypoints_pred.append([kp1, kp2])

            # print('keypoints_pred:', keypoints_pred)
            # print('len(keypoints_pred):', len(keypoints_pred))
            # if len(keypoints_pred) >= 2:
            #     cv2.imshow('pred >= 2?', img)
            #     cv2.waitKey(1)

            if len(keypoints_pred) > 0:
                pred_label = 1

            data['pred_label'] = pred_label
            data['keypoints_pred'] = keypoints_pred
                
            print(img_path, img.shape)

            # json_path = save_paths[mode]
            # print('json_path:', json_path)
            # print('data:', data)
            # print('data_list:', data_list)
            data_list.append(data)

            # print('----------------------------')
        

        data_dict['data_list'] = data_list
        json_path = save_paths[mode]
        print('json_path:', json_path)
        with open(json_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
