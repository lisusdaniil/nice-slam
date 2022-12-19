import argparse
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM
from matplotlib import pyplot as plt
import spatialmath.base as tr

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--dep_u', action='store_true')
    parser.add_argument('--bg_sphr', dest='bg_sphr', action='store_true')
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    
    # turn on background
    args.bg_sphr = True
    slam = NICE_SLAM(cfg, args)
    
    # TEST BACKGROUND RENDERING
    
    device = 'cuda:0'
    # zero the sphere
    grid = torch.zeros_like(slam.shared_c['grid_sphere']).to(device)
    # alternate on azimuth
    for i in range(slam.shared_c['grid_sphere'].shape[3]):
        for j in range(slam.shared_c['grid_sphere'].shape[4]):
            if i % 2 == 0 and j % 2 == 0:
                grid[:,:,:,i,j] = 1.
    
    slam.shared_c['grid_sphere'] = grid

    
    # viewpoint
    nViews = 4
    fig,axs = plt.subplots(2,2)
    angVals = np.linspace(0,3*np.pi/4, nViews)
    for i,ang in enumerate(angVals):
        R_wc = tr.roty(ang)
        t_cw_w = np.zeros((3,1))
        c2w = np.vstack((np.hstack((R_wc, t_cw_w)) , np.array([[0.,0.,0.,1.]])))
        print('Trans Cam to World')
        print(c2w)
        c2w = torch.Tensor(c2w).to(device)
        # Render image
        depth, uncertainty, color = slam.renderer.render_img(
        slam.shared_c,
        slam.shared_decoders,
        c2w,
        device,
        stage='color',
        bg_only=True)
        color_np = color.detach().cpu().numpy()
        a = i % 2
        b = int(np.floor(i / 2))
        print(f"{a},{b}")
        axs[a,b].imshow(color_np, cmap="plasma")
        axs[a,b].set_title(f"Camera Angle:\n{ang*180/np.pi} deg")
        axs[a,b].set_xticks([])
        axs[a,b].set_yticks([])
    plt.tight_layout  
    plt.show()
    
if __name__ == '__main__':
    main()
