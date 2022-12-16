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

    slam = NICE_SLAM(cfg, args)
    
    
    # TEST BACKGROUND RENDERING
    
    device = 'cuda:0'
    # zero the sphere
    grid_azi = torch.zeros_like(slam.shared_c['grid_sphere']).to(device)
    grid_inc = torch.zeros_like(slam.shared_c['grid_sphere']).to(device)
    # alternate on azimuth
    for i in range(slam.shared_c['grid_sphere'].shape[3]):
        if i % 2 == 0:
            grid_inc[:,:,:,i,:] = 1.
        else:
            grid_inc[:,:,:,i,:] = 0.
            
    for i in range(slam.shared_c['grid_sphere'].shape[4]):
        if i % 2 == 0:
            grid_azi[:,:,:,:,i] = 1.
        
    
    # viewpoint
    
    angVals = np.linspace(0,np.pi, 5)
    for ang in angVals:
        R_wc = tr.roty(ang)
        t_cw_w = np.zeros((3,1))
        c2w = np.vstack((np.hstack((R_wc, t_cw_w)) , np.array([[0.,0.,0.,1.]])))
        print('Trans Cam to World')
        print(c2w)
        c2w = torch.Tensor(c2w).to(device)
        # Render image
        slam.shared_c['grid_sphere'] = grid_inc
        depth, uncertainty, color = slam.renderer.render_img(
        slam.shared_c,
        slam.shared_decoders,
        c2w,
        device,
        stage='color',
        bg_only=True)
        color_np = color.detach().cpu().numpy()
        fig,axs = plt.subplots(1,2)
        axs[0].imshow(color_np, cmap="plasma")
        axs[0].set_title(f"inc change: center angle: {ang*180/np.pi} deg")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        # Render image
        slam.shared_c['grid_sphere'] = grid_azi
        depth, uncertainty, color = slam.renderer.render_img(
        slam.shared_c,
        slam.shared_decoders,
        c2w,
        device,
        stage='color',
        bg_only=True)
        color_np = color.detach().cpu().numpy()

        axs[1].imshow(color_np, cmap="plasma")
        axs[1].set_title(f"azi change: center angle: {ang*180/np.pi} deg")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        plt.show()
    
if __name__ == '__main__':
    main()
