import os
import os.path as osp
import argparse
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmseg.apis import init_model

import numpy as np
from math import pi as PI

from common import draw_img_with_mask

def select_work_dir(work_dir, checkpoint):
    print("work_dir:", osp.abspath(work_dir))
    dirs = sorted(os.listdir(work_dir))

    for i, d in enumerate(dirs, 0):
        print("({}) {}".format(i, d))
    d_idx = input("Select directory that you want to load: ")

    path_opt = dirs[int(d_idx)]
    config_dir = osp.abspath(os.path.join(work_dir, path_opt))
    config_path = glob.glob(osp.join(config_dir, '*.py'))[0]

    if checkpoint == 'last':
        with open(osp.join(config_dir, 'last_checkpoint')) as cf:
            pth_path = cf.readline()
    else:
        best_pths =glob.glob(osp.join(config_dir, 'best*.pth'))
        pth_path = best_pths[len(best_pths) - 1]
            
    pth = osp.basename(pth_path)
    pth_path = osp.join(config_dir, pth)

    # print('config_path:', config_path)
    # print('pth_path:', pth_path)

    return config_path, pth_path

class Warping(nn.Module):
    def __init__(self, direction: str='cart2polar', n_beams: int=512, scale: float=0.5):
        super.__init__()

        self.direction = direction
        self.n_beams = n_beams
        self.scale = scale

    def forward(self, x):
        if self.direction == 'cart2polar':
            cart = x
            rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
            theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
            theta = theta + (theta < 0).type_as(theta) * (2 * PI)
            polar = torch.cat([rho, theta], dim=-1)
            out = polar

        out = out
        return out

    def backward(self):
        self.loss.backward()



if __name__ == '__main__':
    work_dir = '../../../data/mmsegmentation/work_dirs'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', type=str, default=work_dir,
        help='specify working directory of trainined model')
    parser.add_argument('--checkpoint', type=str, default='best', choices=['last', 'best'],
        help='select which chekpoint will be chosen [last|best]')
    parser.add_argument('--input_size', type=int, default=512,
        help='the size of input image')
    parser.add_argument('--verbose', action='store_true',
        help='show summary of the model')
    
    args = parser.parse_args()

    config_path, pth_path = select_work_dir(args.work_dir, args.checkpoint)
    configname, _ = osp.splitext(osp.basename(config_path))
    cfg = Config.fromfile(config_path)

    # init model and load checkpoint
    device = 'cuda:0'
    print(f'Initializing model with {config_path} and {pth_path}')
    net = init_model(config_path, pth_path)

    # print('net:', net)
    net = net.to(device)
    
    warping = Warping()
    warping = warping.to(device)


    test_dir = osp.join('..', cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.data_prefix.img_path)
    annot_dir = osp.join('..', cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.data_prefix.seg_map_path)

    case_list = os.listdir(test_dir)

    if '.png' in case_list[0]:
        case_list = ['.']
    
    for case in case_list:
        case_dir = osp.join(test_dir, case)
        file_list = os.listdir(case_dir)
        annot_case_dir = osp.join(annot_dir, case)


        # save_case_dir = osp.join(save_dir, case)
        # os.makedirs(save_case_dir, exist_ok=True)
        for fp in file_list:
            # fp = file_list[4]
            img_path = osp.join(case_dir, fp)
            annot_path = osp.join(annot_case_dir, fp)
            # result_path= osp.join(save_case_dir, fp)
            # result_path= osp.join(save_dir, fp)
            print(f'Inference on {img_path}')

            flat_np =cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            annot_np = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE)
            annot_np = annot_np == np.max(annot_np) # Make int data type to boolean

            flat_t = torch.from_numpy(flat_np).float().to(device)
            flat_t = flat_t.reshape((1, 1, flat_t.shape[0], flat_t.shape[1]))
            flat_t = flat_t / 255.0

            round_t = warping(flat_t)
            round_np = round_t.squeeze().to('cpu').detach().numpy()

            cv2.imshow('results', round_np)
            cv2.waitKey()


            # # out_t = net(flat_t, mode='predict')
            # # mask_t = out_t[0].pred_sem_seg.data

            
            # out_t = net(flat_t, mode='tensor')
            # if '-ce' in pth_path:
            #     print('multi class')
            #     mask_t = torch.argmax(out_t, dim=1)
            # else:
            #     print('binary class')
            #     # print('out_t:', out_t)
            #     # out_t = F.sigmoid(out_t)
                
            #     out_t = out_t.sigmoid()
            #     # print('sigmoid out_t', out_t)
            #     mask_t = out_t >= 0.3
            # print('mask_t.shape:', mask_t.shape)

            # nz = mask_t.nonzero()
            # print('nz_.shape:', nz.shape)
            # print('nz:', nz)
            # x_ = nz[:, 3]
            # print('x_.shape:', x_.shape)
            # print('x_:', x_)
            # x_ = x_.unique()
            # print('unique x_', x_)
            # mask_t[:, :, :, x_] = 1
            
            # pred_np = mask_t.squeeze().to('cpu').detach().numpy().astype(np.bool_)
            # pred_masked = draw_img_with_mask(flat_np, pred_np, color=(0, 255, 0), alpha=0.2)     
            # gt_masked = draw_img_with_mask(flat_np, annot_np, color=(0, 0, 255), alpha=0.2)

            # results = np.concatenate([pred_masked, gt_masked], axis=1)

            cv2.imshow('results', results)
            cv2.waitKey()
