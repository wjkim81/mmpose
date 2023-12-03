import os
import os.path as osp
import cv2
import numpy as np

def draw_img_with_mask(img, mask, color=(255,255,255), alpha=0.8):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img.astype(np.float32)
    img_draw = img.copy()
    img_draw[mask] = color
    out = img * (1 - alpha) + img_draw * alpha
    
    return out.astype(np.uint8)

def select_checkpoint(work_dir):
    print("work_dir:", osp.abspath(work_dir))
    dirs = sorted(os.listdir(work_dir))

    for i, d in enumerate(dirs, 0):
        print("({}) {}".format(i, d))
    d_idx = input("Select checkpoint that you want to load: ")

    path_opt = dirs[int(d_idx)]
    chosen_checkpoint = osp.abspath(os.path.join(work_dir, path_opt))
    
    print(f'loaded {chosen_checkpoint}')

    return chosen_checkpoint