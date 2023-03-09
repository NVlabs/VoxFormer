import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti_util
import numpy as np


def project_disp_to_depth(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--calib_dir', type=str, default='kitti_dir/dataset/sequences/00')
    parser.add_argument('--depth_dir', type=str, default='depth_dir/sequences/00')
    parser.add_argument('--save_dir', type=str, default='lidar_dir/sequences/00')
    parser.add_argument('--max_high', type=int, default=80)
    args = parser.parse_args()

    assert os.path.isdir(args.depth_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    depths = [x for x in os.listdir(args.depth_dir) if x[-3:] == 'npy' and 'std' not in x]
    depths = sorted(depths)

    for fn in depths:
        predix = fn[:-4]
        # predix = fn[:-8]
        # calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib_file = '{}/{}.txt'.format(args.calib_dir, 'calib')
        calib = kitti_util.Calibration(calib_file)
        depth_map = np.load(args.depth_dir + '/' + fn)

        lidar = project_disp_to_depth(calib, depth_map, args.max_high)
        # pad 1 in the indensity dimension
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Depth {}'.format(predix))
