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



def select_work_dir(work_dir, checkpoint):
    print("work_dir:", osp.abspath(work_dir))
    dirs = sorted(os.listdir(work_dir))

    for i, d in enumerate(dirs, 0):
        print("({}) {}".format(i, d))
    d_idx = input("Select directory that you want to load: ")

    path_opt = dirs[int(d_idx)]
    chosen_dir = osp.abspath(os.path.join(work_dir, path_opt))
    config_path = glob.glob(osp.join(chosen_dir, '*.py'))[0]

    if checkpoint == 'last':
        with open(osp.join(chosen_dir, 'last_checkpoint')) as cf:
            pth_path = cf.readline()
    else:
        with open(osp.join(chosen_dir, 'lest_checkpoint')) as cf:
            pth_path = cf.readline()
    pth = osp.basename(pth_path)
    pth_path = osp.join(chosen_dir, pth)

    print('config_path:', config_path)
    print('pth_path:', pth_path)

    return config_path, pth_path

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
    parser.add_argument('--thr', type=float, default=0.5,
        help='the threshold of keypoiont score')
    parser.add_argument('--correct_range', type=int, default=15,
        help='the range which include the correct prediction')
    parser.add_argument('--round2flat', action='store_true',
        help='convert round to flat view')

    args = parser.parse_args()

    pred_color = (255, 0, 0)
    gt_color = (0, 255, 0)

    config_path, pth_path = select_work_dir(args.work_dir, args.checkpoint)
    configname = osp.splitext(osp.basename(config_path))[0]
    cfg = Config.fromfile(config_path)
    # print(cfg.test_dataloader)

    # # init model and load checkpoint
    print(f'Initializing model with {config_path} and {pth_path}')
    # model = init_model(config_path, pth_path, device='cuda')
    inferencer = MMPoseInferencer(pose2d=config_path, pose2d_weights=pth_path)

    test_dir = osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.data_prefix.img)
    annot_path = osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.ann_file)
    if args.round2flat:
        test_dir.replace('round', 'flat')
        annot_path.replace('round', 'flat')

    save_dir = osp.join(rf'../../data/mmpose/results/{cfg.test_dataloader.dataset.type}', configname)
    os.makedirs(save_dir, exist_ok=True)

    with open(annot_path, 'r') as json_file:
        annot_data = json.load(json_file)

    image_list = annot_data['images']
    annot_list = annot_data['annotations']

    annot_dict = {}
    for ann in annot_list:
        
        img_id = ann['image_id']

        if not img_id in annot_dict:
            annot_dict[img_id] = [ann]
        else:
            annot_dict[img_id].append(ann)
        
    # print('image_list:', image_list)
    for img_data in image_list:
        file_name = img_data['file_name']
        img_id = img_data['id']

        img_path = osp.join(test_dir, file_name)
        img_fn = file_name.replace('/', '-')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result_generator = inferencer(img_path)
        # inferencer.visualize()
        result = next(result_generator)
        # print(result_generator)
        
        predictions = result['predictions']

        print('**************************')
        # print('predictions:', predictions)

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
        keypoints_pred = []
        keypoints_gt = []
        for preds in predictions:
        #     print('preds:', preds)
            for p in preds:
        #         print('p', p)
                keypoints_pred = p['keypoints']
                keypoint_scores = p['keypoint_scores']
                # print('keypoints:', keypoints_pred)
                # print('keypoint_scores:', keypoint_scores)

                keypoint_exist = False
                for ks in keypoint_scores:
                    if ks > 0.5: keypoint_exist = True

                if keypoint_exist:
                    kp1 = [int(x + 0.5) for x in keypoints_pred[0]]
                    kp2 = [int(x + 0.5) for x in keypoints_pred[1]]
                    print('keypoint_scores:', keypoint_scores)
                    print('predictions:', kp1, kp2)
                    cv2.circle(img, kp1, 2, pred_color, -1)
                    cv2.circle(img, kp1, 10, pred_color, 1)
                    cv2.circle(img, kp2, 2, pred_color, -1)
                    cv2.circle(img, kp2, 10, pred_color, 1)
                    keypoints_pred.append((kp1, kp2))                    

                

                # for i, ks in enumerate(keypoint_scores):
                #     if ks > args.thr:
                #         kp1 = [int(x + 0.5) for x in keypoints_pred[i][0]]
                #         kp2 = [int(x + 0.5) for x in keypoints_pred[i][1]]
                #         print('keypoint_scores:', keypoint_scores)
                #         print('predictions:', kp1, kp2)
                #         cv2.circle(img, kp1, 2, pred_color, -1)
                #         cv2.circle(img, kp2, 2, pred_color, -1)
                #         keypoints_pred.append((kp1, kp2))


        if img_id in annot_dict:
            annots = annot_dict[img_id]

            for ann in annots:
                keypoint_gt = ann['keypoints']
                print(keypoint_gt)
                keypoints = get_keypoints(keypoint_gt)
                print(keypoints)
                keypoints_gt.append(keypoints)
                cv2.circle(img, keypoints[0], 1, gt_color, 1)
                cv2.circle(img, keypoints[0], 10, gt_color, 1)
                cv2.circle(img, keypoints[1], 1, gt_color, 1)
                cv2.circle(img, keypoints[1], 10, gt_color, 1)


        if args.round2flat:
            print('convert keypoint points in round view to flat view')
        n_gt = len(keypoints_gt)
        n_pred = len(keypoints_pred)

        for kp_pair in keypoints_gt:
            gt_interval = sorted([x[0] for x in kp_pair])
            print('gt_interval:', gt_interval)

        for kp_pair in keypoints_pred:
            pred_interval = sorted([x[0] for x in kp_pair])
            print('pred_interval:', pred_interval)

        cv2.imshow('image', img)
        cv2.waitKeyEx(1)
        print(img_path, img.shape)


        result_path = osp.join(save_dir, file_name.replace('/', '-'))
        print(osp.abspath(result_path))
        cv2.imwrite(result_path, img)
        print('----------------------------')

        

                               
        # if img_id in annot_dict:
            # print(annot_dict[img_id]