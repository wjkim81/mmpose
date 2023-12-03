# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import glob

import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner


def parse_args(work_dir='../../data/mmpose/work_dirs'):
    parser = argparse.ArgumentParser(
        description='MMPose test (and eval) model')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint',
        default=None,
        type=str,
        help='checkpoint file')

    parser.add_argument('--work-dir', type=str, default=work_dir, 
        help='the directory to save evaluation results')
    
    parser.add_argument('--load',
        type=str,
        default='best',
        choices=['last', 'best'],
        help='select which chekpoint will be chosen [last|best]')
    
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    # -------------------- work directory --------------------
    # work_dir is determined in this priority: CLI > segment in file > filename
    # if args.work_dir is not None:
    #     # update configs according to CLI args if args.work_dir is not None
    #     cfg.work_dir = args.work_dir
    # elif cfg.get('work_dir', None) is None:
    #     # use config filename as default work_dir if cfg.work_dir is None
    #     cfg.work_dir = osp.join('./work_dirs',
    #                             osp.splitext(osp.basename(args.config))[0])

    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    # -------------------- Other arguments --------------------
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg

def select_work_dir(args):
    work_dir = args.work_dir
    load = args.load
    print("work_dir:", osp.abspath(work_dir))
    dirs = sorted(os.listdir(work_dir))

    for i, d in enumerate(dirs, 0):
        print("({}) {}".format(i, d))
    d_idx = input("Select directory that you want to load: ")

    path_opt = dirs[int(d_idx)]
    chosen_dir = osp.abspath(os.path.join(work_dir, path_opt))
    config_path = glob.glob(osp.join(chosen_dir, '*.py'))[0]

    if args.checkpoint is None:
        if load == 'last':
            with open(osp.join(chosen_dir, 'last_checkpoint')) as cf:
                pth_path = cf.readline()
        else:
            best_pths =glob.glob(osp.join(chosen_dir, 'best*.pth'))
            # print(best_pths)
            pth_path = best_pths[len(best_pths) - 1]
                
        pth = osp.basename(pth_path)
        pth_path = osp.join(chosen_dir, pth)
    
        args.checkpoint = pth_path

    # print('config_path:', config_path)
    # print('pth_path:', args.checkpoint)

    return config_path, pth_path


def main():
    work_base_dir = r'../../data/mmpose/work_dirs'
    args = parse_args(work_dir=work_base_dir)

    config_path, checkpoint_path = select_work_dir(args)
    configname = osp.splitext(osp.basename(config_path))[0]


    # load config
    # cfg = Config.fromfile(args.config)
    cfg = Config.fromfile(config_path)
    cfg = merge_args(cfg, args)

    cfg.checkpoint = checkpoint_path
    cfg.work_dir = osp.join(work_base_dir, configname)
    print('cfg.work_dir:', cfg.work_dir)


    # build the runner from config
    runner = Runner.from_cfg(cfg)

    if args.out:

        class SaveMetricHook(Hook):

            def after_test_epoch(self, _, metrics=None):
                if metrics is not None:
                    mmengine.dump(metrics, args.out)

        runner.register_hook(SaveMetricHook(), 'LOWEST')

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
