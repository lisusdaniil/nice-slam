import os
import numpy as np
import sys
import argparse
import random
import torch
from liegroups.torch import SO3

sys.path.append('../../')
from src import config
from src.utils.datasets import get_dataset

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def gen_imu(cfg, args):
    # Load in dataset with gt poses
    dtset = get_dataset(cfg, args, cfg['scale'])

    # Arbitrarily set dt since we have no timesteps anywhere
    dt = 0.01
    dt_list = dt * torch.ones(dtset.n_img-1)

    # Initialize measurement vectors
    gyro_meas = torch.zeros((dtset.n_img-1, 3))
    vel_meas = torch.zeros((dtset.n_img-1, 3))

    for ii in range(0, dtset.n_img-1):
        if dtset.timestamps is not None:
            dt = dtset.timestamps[ii+1] - dtset.timestamps[ii]
            dt_list[ii] = dt

        # Get current state
        c2w_ii = dtset.poses[ii]
        C_ab_ii = SO3.from_matrix(c2w_ii[0:3, 0:3], normalize=True)
        r_a_ii = c2w_ii[0:3, 3:4]

        # Get next future state
        c2w_jj = dtset.poses[ii+1]
        C_ab_jj = SO3.from_matrix(c2w_jj[0:3, 0:3], normalize=True)
        r_a_jj = c2w_jj[0:3, 3:4]

        # Compute velocity in F_a
        v_a_ii = (r_a_jj - r_a_ii)/dt
        # Resolve velocity in F_b_ii and save
        vel_meas[ii,:] = C_ab_ii.inv().dot(v_a_ii.squeeze())

        # Get relative change between b_ii and b_jj body frame in R^n, 
        # divide by dt to get angular vel
        gyro_meas[ii,:] = (C_ab_ii.inv().dot(C_ab_jj)).log()/dt
        
    # Add noise
    vel_meas = vel_meas + np.random.multivariate_normal(np.zeros(3), args.vel_noise*np.eye(3), dtset.n_img-1)
    gyro_meas = gyro_meas + np.random.multivariate_normal(np.zeros(3), args.gyro_noise*np.eye(3), dtset.n_img-1)

    # Generate dead reckoned poses using noisy imu data
    dr_poses = torch.zeros((dtset.n_img, 4, 4))
    dr_poses[0, :, :] = dtset.poses[0]
    # Initialize dr error vector
    dr_err = torch.zeros((dtset.n_img-1, 3))
    for ii in range(0, dtset.n_img-1):
        # Get current state
        c2w_ii = dr_poses[ii]
        C_ab_ii = SO3.from_matrix(c2w_ii[0:3, 0:3], normalize=True)
        r_a_ii = c2w_ii[0:3, 3:4]

        # Integrate velocity to get position
        r_a_jj = r_a_ii + dt_list[ii] * C_ab_ii.dot(vel_meas[ii,:].unsqueeze(0).float()).unsqueeze(1)

        # Integrate angular velocity to get rotation
        C_ab_jj = C_ab_ii.dot(SO3.exp(dt * gyro_meas[ii,:].unsqueeze(0).float()))

        # Save new pose
        dr_poses[ii+1, 0:3, 0:3] = C_ab_jj.mat
        dr_poses[ii+1, 0:3, 3:4] = r_a_jj

    # Save all data in imu dict
    imu_data = {'vel': vel_meas, 'gyro': gyro_meas, 'dt': dt_list, 'vel_std': args.vel_noise, 'gyro_std': args.gyro_noise}

    # Save imu dict to
    torch.save(imu_data, args.imu_dir + '/imu_data')
    path = os.path.join(args.imu_dir, 'ckpts/')
    # Create path if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
            'gt_c2w_list': dtset.poses,
            'estimate_c2w_list': dr_poses,
            'idx': dtset.n_img-1
        }, path+'dr.tar', _use_new_zipfile_serialization=False)

def main():
    setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the IMU generation.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--vel_noise', type=float,
                        help='linear velocity noise std value', default=0.01)
    parser.add_argument('--gyro_noise', type=float,
                        help='angular velocity noise std value', default=0.01)
    args = parser.parse_args()
    cfg = config.load_config(args.config, '../../configs/nice_slam.yaml')
    args.input_folder = '../../' + cfg['data']['input_folder']
    args.output = cfg['data']['output']

    # Make dir for imu data
    args.imu_dir = os.path.join(args.input_folder, 'imu')

    os.makedirs(args.imu_dir, exist_ok=True)

    gen_imu(cfg, args)

if __name__ == '__main__':
    main()
