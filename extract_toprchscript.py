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

from octmodules.utils import mm_select_work_dir


class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    def forward(self, x):
        with torch.no_grad():
            out = self.model.test_step(x)
        # out = out
        return out
    

if __name__ == '__main__':
    work_dir = '../../data/mmpose/work_dirs'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', type=str, default=work_dir,
        help='specify working directory of trainined model')
    # parser.add_argument('--no_wrapper', action='store_true',
    #     help='save model without wraooer')    
    parser.add_argument('--checkpoint', type=str, default='best', choices=['last', 'best'],
        help='select which chekpoint will be chosen [last|best]')
    parser.add_argument('--input_size', type=int, default=512,
        help='the size of input image')
    parser.add_argument('--verbose', action='store_true',
        help='show summary of the model')
    
    args = parser.parse_args()

    config_path, pth_path = mm_select_work_dir(args.work_dir, args.checkpoint)
    configname, _ = osp.splitext(osp.basename(config_path))
    cfg = Config.fromfile(config_path)

    # init model and load checkpoint
    device = 'cuda'
    print(f'Initializing model with {config_path} and {pth_path}')
    model = init_model(config_path, pth_path, device=device)

    # print('model:', model)

    # if args.no_wrapper:
    #     fn = f'{configname}-no_wrapped'
    # else:
    #     fn = f'{configname}-wrapped'

    fn = f'{configname}-wrapped'
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    # print('pipeline:', pipeline)
    img = np.random.rand(512, 512)
    data_info = dict(img=img)
    data_info.update(model.dataset_meta)
    print('data_info:', data_info)

    data = pipeline(data_info)
    print('data:', data)
    batch = pseudo_collate([data])
    net = Model(model)

    # x = torch.rand(1, 1, args.input_size, args.input_size).to('cuda')
    x = torch.rand(1, 1, args.input_size, args.input_size)

    if args.verbose:
        print(pytorch_model_summary.summary(net, x, max_depth=None, show_parent_layers=True, show_input=True))
    
    save_dir = osp.join(args.work_dir, '..', 'z-torchscript_models')
    os.makedirs(save_dir, exist_ok=True)
    
    '''
    Pytorch model should saved with model definition
    '''
    # save_pth_path = osp.join(save_dir, f'{fn}.pth')
    # # torchsummary.summary(net, input_size=x.shape[1:])
    # print(f'saving torch model at {save_pth_path}')
    # torch.save(net, save_pth_path)

    # save for torch.jit moduel
    save_jit_path = osp.join(save_dir, f'{fn}-torchinput{args.input_size}x{args.input_size}.jit.pth')
    module = torch.jit.trace(net.forward, x)
    print(f'saving torch.jit module at {save_jit_path}')
    torch.jit.save(module, save_jit_path)
