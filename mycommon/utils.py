import os
import os.path as osp
import glob
import platform
import numpy as np
from skimage.color import gray2rgb

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

            # print('pth_path:', pth_path)

            if platform.system() == 'Windows':
                # print('Windows')
                pth_path = pth_path.replace('/', '\\')
            elif platform.system() == 'Linux':
                # print('Linux')
                pth_path = pth_path.replace('\\', '/')
            else:
                raise OSError('Which operating system are you using?')

            # print('base pth_path:', pth_path)
            pth = osp.basename(pth_path)
            pth_path = osp.join(chosen_dir, pth)
    else:
        with open(osp.join(chosen_dir, 'best_checkpoint')) as cf:
            pth_path = glob.glob(osp.join(chosen_dir, 'best*.pth'))[0]

    print('config_path:', config_path)
    print('pth_path:', pth_path)

    return config_path, pth_path      
def draw_img_with_mask(img, mask, color=(255,255,255), alpha=0.8):
    if img.ndim == 2:
        img = gray2rgb(img)
    img = img.astype(np.float32)
    img_draw = img.copy()
    img_draw[mask] = color
    out = img * (1 - alpha) + img_draw * alpha
    
    return out.astype(np.uint8)
