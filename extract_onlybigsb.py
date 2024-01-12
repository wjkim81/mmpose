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

    pred_color = (255, 0, 0)
    gt_color = (0, 255, 0)
    guidewire_color = (0, 0, 255)

    config_path, pth_path = mm_select_work_dir(args.work_dir, args.checkpoint)
    configname = osp.splitext(osp.basename(config_path))[0]
    cfg = Config.fromfile(config_path)
    # print(cfg.test_dataloader)


    device = 'cuda'

    guidewire_seg_chk_path = '../../data/mmsegmentation/z-torchscript_models/unet-s5-d16_fcn-bce_4xb4-1280k_octflatguidewires-random_resize512x512-crop256x256-no_wrapped-torchinput512x512.jit.pth'
    guidewire_net = torch.jit.load(guidewire_seg_chk_path, map_location=torch.device(device))
    guidewire_net.eval()

    
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
        'train': osp.join(cfg.train_dataloader.dataset.data_root, 'bigsb-' + cfg.train_dataloader.dataset.ann_file).replace('round', 'flat'),
        'test': osp.join(cfg.test_dataloader.dataset.data_root, 'bigsb-' + cfg.test_dataloader.dataset.ann_file).replace('round', 'flat')
    }
    bigsb_round_annot_path_dict = {
        'train': osp.join(cfg.train_dataloader.dataset.data_root, 'bigsb-' + cfg.train_dataloader.dataset.ann_file).replace('flat', 'round'),
        'test': osp.join(cfg.test_dataloader.dataset.data_root, 'bigsb-' + cfg.test_dataloader.dataset.ann_file).replace('flat', 'round')
    }

    round2flat = False
    if 'round' in configname:
        round2flat = True

    save_dir = osp.join(rf'../../data/mmpose/results-onlybigsb/{cfg.test_dataloader.dataset.type}', f'{configname}-kp_range{args.keypoint_range}-gw_thr{args.guidewire_threshold}')
    # save_dir_dict = {
    #     'train': osp.join(rf'../../data/mmpose/results-onlybigsb/{cfg.train_dataloader.dataset.type}', f'{configname}-kp_range{args.keypoint_range}-gw_thr{args.guidewire_threshold}'),
    #     'test': osp.join(rf'../../data/mmpose/results-onlybigsb/{cfg.test_dataloader.dataset.type}', f'{configname}-kp_range{args.keypoint_range}-gw_thr{args.guidewire_threshold}')
    # }
    # os.makedirs(save_dir, exist_ok=True)



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

        image_round_list = annot_round_data['images']
        annot_round_list = annot_round_data['annotations']

        bigsb_flat_annot_dict = {
            'images': image_flat_list,
            'annotations': []
        }
        bigsb_round_annot_dict = {
            'images': image_round_list,
            'annotations': []
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
        # print('image_list:', image_list)

        # Group images by case in dictionary
        cases = {}
        for img_data in image_flat_list:
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
            guidewires = {}

            included_index = []
            keypoints_flat_annot ={}
            keypoints_round_annot ={}
            
            for i, img_data in enumerate(case_images):
                file_name = img_data['file_name']
                img_id = img_data['id']

                img_path = osp.join(img_dir, file_name)

                if round2flat:
                    flat_path = img_path.replace('round', 'flat')
                else:
                    flat_path = img_path

                flat_img = cv2.imread(flat_path, cv2.IMREAD_UNCHANGED)

                flat_np = flat_img.copy()
                if flat_img.ndim == 2:
                    flat_np = np.expand_dims(flat_img, 2)
                flat_np = flat_np / 255
                flat_np = np.expand_dims(flat_np.transpose(2, 0, 1), 0)
                flat_t = torch.from_numpy(flat_np).float()
                flat_t = flat_t.to(device)

                # Segment guidewires
                p = guidewire_net(flat_t)
                p = p.sigmoid()
                # print('p.shape:', p.shape)
                m = p > args.threshold
                nz = m.nonzero()
                x_ = nz[:, 3].unique()
                x_range = x_.detach().to('cpu').numpy()

                guidewires[i] = x_range
                guidewires_range = x_range
                # m = m.float()

                # Keypoints predictions in flat view
                result_generator = inferencer(img_path)
                result = next(result_generator)
                predictions = result['predictions']


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
                            include_keypoint = True

                            kp1 = [int(x + 0.5) for x in kp_pred[0]]
                            kp2 = [int(x + 0.5) for x in kp_pred[1]]
                            kp_pred = (kp1, kp2)
                            if round2flat:
                                print('Warping predicted keypoints from round to flat')
                                print('kp_pred:', kp_pred)
                                kp_pred = warp_keypoints_to_cart(kp_pred, flat_img.shape)

                            if kp1[0] > kp2[0]:
                                tmp = kp2
                                kp2 = kp1
                                kp1 = tmp
                                kp_pred = (kp1, kp2)

                            pred_x_range = list(range(kp1[0], kp2[0]))
                            pred_length = len(pred_x_range)
                            if pred_length > 256:
                                kp_pred = kp_pred[1], kp_pred[0]
                            # remove small length of keypoints
                            # if pred_length < args.keypoint_range:
                            #     print(f'{i} smal prediction keypoint - length: {pred_length}')
                            #     include_keypoint = False

                            # check whether kp_pred in guidewire shadow
                            duplicate_len = len(set(pred_x_range) & set(guidewires_range))
                            if pred_length == 0:
                                pred_length = 1
                            duplicate_ratio = duplicate_len / pred_length
                        
                            if duplicate_ratio > args.guidewire_threshold:
                                print('n duplicate:', duplicate_len)
                                print('n duplicate_ratio:', duplicate_ratio)
                                print('pred_x_range:', pred_x_range)
                                print('guidewires_range:', guidewires_range)
                                print('duplicating with guidewires')
                                include_keypoint = False
                                
                            
                            if include_keypoint:
                                keypoints_pred[i] = kp_pred

                # Keypoints ground truth in flat view
                gt_x_range = []
                if img_id in annot_flat_dict:
                    annots_flat = annot_flat_dict[img_id]
                    annots_round = annot_round_dict[img_id]

                    for ann_flat, ann_round in zip(annots_flat, annots_round):
                        include_keypoint = True

                        keypoint_gt = ann_flat['keypoints']
                        keypoints = get_keypoints(keypoint_gt)

                        kp1, kp2 = keypoints
                        
                        if kp1[0] > kp2[0]:
                            tmp = kp2
                            kp2 = kp1
                            kp1 = tmp
                            keypoints = (kp1, kp2)

                        include_keypoint = True

                        gt_x_range = list(range(kp1[0], kp2[0]))
                        gt_length = len(gt_x_range)

                        # remove small length of keypoints
                        if gt_length > 256:
                            keypoints = keypoints[1], keypoints[0]
                        # if gt_length < args.keypoint_range:
                        #     print(f'{i} small gt keypoint - length: {gt_length}')
                        #     include_keypoint = False

                        # print((i, keypoints))
                        if include_keypoint:
                            keypoints_gt[i] = keypoints
                            included_index.append(i)
                            keypoints_flat_annot[i] = ann_flat
                            keypoints_round_annot[i] = ann_round
                
                img_volume.append(np.expand_dims(flat_img, axis=0))

            print('included_index:', included_index)

            continuous_list = group_continuous(included_index)
            print('continuous_list:', continuous_list)

            print('keypoints_gt:', keypoints_gt)

            for cl in continuous_list:
                if len(cl) > 3:
                    for idx in cl:
                        # keypoints = keypoints_gt[idx]
                        bigsb_flat_annot_dict['annotations'].append(keypoints_flat_annot[idx])
                        bigsb_round_annot_dict['annotations'].append(keypoints_round_annot[idx])
                else:
                    for idx in cl:
                        print(idx)
                        if idx in keypoints_gt:
                            keypoints_gt.pop(idx)

            print('only big sb keypoints_gt:', keypoints_gt)
            
            img_volume = np.concatenate(img_volume, axis=0)
            print('img_volume.shape:', img_volume.shape)
            img_volume_std = np.std(img_volume, axis=1)
            print('img_volume_std.shape:', img_volume_std.shape)
            # print('img_volume_std.dtype:', img_volume_std.dtype)

            v_max = img_volume_std.max()
            v_min = img_volume_std.min()

            img_volume_std = (img_volume_std - v_min)/ (v_max - v_min)
            img_volume_std = 255 * img_volume_std
            img_volume_std = img_volume_std.astype(np.uint8)

            nf, resolution = img_volume_std.shape

            img_volume_std = cv2.cvtColor(img_volume_std, cv2.COLOR_GRAY2RGB)

            # draw mask of each feature on projected of image volume
            pred_mask = np.zeros((nf, resolution), dtype=np.bool_)
            gt_mask = np.zeros((nf, resolution), dtype=np.bool_)
            guidewire_mask = np.zeros((nf, resolution), dtype=np.bool_)
            

            # Draw masking with keypoints saved dictionary
            # Here remove one frame keypoints
            for i in range(nf):
                if i in keypoints_gt:
                    kp_gt = keypoints_gt[i]
                    # pt1 = (kp_gt[0][1], i)
                    # pt2 = (kp_gt[1][1], i)
                    print(f'{i} = {kp_gt[0][0]} - {kp_gt[1][0]}')
                    gt_mask[i, kp_gt[0][0]:kp_gt[1][0]] = 1
                
                if i in keypoints_pred:
                    kp_pred = keypoints_pred[i]
                    # pt1 = (kp_pred[0][1], i)
                    # pt2 = (kp_pred[1][1], i)
                    pred_mask[i, kp_pred[0][0]:kp_pred[1][0]] = 1
                        
                if i in guidewires:
                    x_ = guidewires[i]
                    guidewire_mask[i, x_] = 1


            # pred_color = (255, 0,~ 0)
            # gt_color = (0, 255, 0)
            # guidewire_color = (0, 0, 255)
            gt_volume = draw_img_with_mask(img_volume_std, guidewire_mask, color=guidewire_color, alpha=0.2)
            gt_volume = draw_img_with_mask(gt_volume, gt_mask, color=gt_color, alpha=0.5)
            pred_volume = draw_img_with_mask(img_volume_std, guidewire_mask, color=guidewire_color, alpha=0.2)
            pred_volume = draw_img_with_mask(pred_volume, pred_mask, color=pred_color, alpha=0.5)

            h, w, c = pred_volume.shape
            # gt_volume[:, :2] = (0, 0, 0)

            vertical_line = np.zeros((h, 2, c), dtype=pred_volume.dtype)
            img_vis = np.concatenate([pred_volume, vertical_line, gt_volume], axis=1)
            
            # resize for visualization
            img_vis = cv2.resize(img_vis, (2*resolution, 5*nf), interpolation=cv2.INTER_CUBIC)
            cv2.imshow(f'mask-{args.keypoint_range}', img_vis)
            cv2.waitKey(1)

            result_path = osp.join(save_dir, f'{case}.png')
            print(f'saving image volume making on {result_path}')

            cv2.imwrite(result_path, img_vis)

        
        print(f'Saving big side branch extracted annotation to {dst_flat_annot_path}')
        print(f'Saving big side branch extracted annotation to {dst_round_annot_path}')
        with open(dst_flat_annot_path, 'w') as annot_json:
            json.dump(bigsb_flat_annot_dict, annot_json, indent=4)
        with open(dst_round_annot_path, 'w') as annot_json:
            json.dump(bigsb_round_annot_dict, annot_json, indent=4)
