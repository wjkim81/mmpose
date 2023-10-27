import os
import os.path as osp
import glob
import argparse
import imageio
import numpy as np

from mmengine.config import Config

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

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


if __name__ == '__main__':
    work_dir = '../../data/mmpose/work_dirs'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', type=str, default=work_dir,
        help='specify working directory of trainined model')
    parser.add_argument('--checkpoint', type=str, default='last', choices=['last', 'best'],
        help='select which chekpoint will be chosen [last|best]')

    args = parser.parse_args()

    config_path, pth_path = select_work_dir(args.work_dir, args.checkpoint)
    configname = osp.splitext(osp.basename(config_path))[0]
    cfg = Config.fromfile(config_path)
    # print(cfg.test_dataloader)

    # # init model and load checkpoint
    print(f'Initializing model with {config_path} and {pth_path}')
    model = init_model(config_path, pth_path, device='cuda')

    test_dir = osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.data_prefix.img)
    # annot_dir = osp.join(cfg.test_dataloader.dataset.data_root, cfg.test_dataloader.dataset.data_prefix.seg_map_path)
    save_dir = osp.join(rf'../../data/mmpose/results/{cfg.test_dataloader.dataset.type}', configname)
    os.makedirs(save_dir, exist_ok=True) 
