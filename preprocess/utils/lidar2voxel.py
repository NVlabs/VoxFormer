# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os

import numpy as np
from tqdm.contrib.concurrent import process_map

from collections import deque
import shutil
from numpy.linalg import inv
import struct
import time
import mapping
 

def pack(array):
  """ convert a boolean array into a bitwise array. """
  array = array.reshape((-1))

  #compressing bit flags.
  # yapf: disable
  compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
  # yapf: enable

  return np.array(compressed, dtype=np.uint8)

def parse_calibration(filename):
  """ read calibration file with given filename

      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib


def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses

def parallel_work_sequence(f):

  # read scan and labels, get pose
  scan_filename = os.path.join(input_folder, f)
  scan = np.fromfile(scan_filename, dtype=np.float32)

  scan = scan.reshape((-1, 4))

  # convert points to homogenous coordinates (x, y, z, 1)
  points = np.ones((scan.shape))
  points[:, 0:3] = scan[:, 0:3]
  remissions = scan[:, 3]

  # prepare single numpy array for all points that can be written at once.
  num_concat_points = points.shape[0]

  # num_concat_points += sum([past["points"].shape[0] for past in history])
  concated_points = np.zeros((num_concat_points * 4), dtype = np.float32)
  # concated_labels = np.zeros((num_concat_points), dtype = np.uint32)

  start = 0
  concated_points[4 * start:4 * (start + points.shape[0])] = scan.reshape((-1))
  # concated_labels[start:start + points.shape[0]] = labels
  start += points.shape[0]

  if float(os.path.splitext(f)[0])%5==0:
  # if True:
    #------------------------------------------------------------------------------------------------------------------------  
    #------------------------------------------------------------------------------------------------------------------------  
    # (1). visibility (voxel) generation
    visibility_maps = []
    # origins = np.zeros((1,4))
    origins = np.array([[0, 0, 0, 1]])
    map_dims = [256, 256, 32]
    voxel_size = (0.2, 0.2, 0.2)
    area_extents = np.array([[0, 51.2], [-25.6, 25.6], [-2., 4.4]])
    
    pc_range = [area_extents[0,0], area_extents[1,0], area_extents[2,0], area_extents[0,1], area_extents[1,1], area_extents[2,1]]
    pts = concated_points.reshape(-1,4)
    filter_idx = np.where((area_extents[0, 0] < pts[:, 0]) & (pts[:, 0] < area_extents[0, 1]) & (area_extents[1, 0] < pts[:, 1]) & (pts[:, 1] < area_extents[1, 1]) & (area_extents[2, 0] < pts[:, 2]) & (pts[:, 2] < area_extents[2, 1]))[0]
    pts = pts[filter_idx]
      
    visibility_maps.append(mapping.compute_logodds_dp(pts, origins[[0],:3], pc_range, range(pts.shape[0]), 0.2)) #, lo_occupied, lo_free
    visibility_maps = np.asarray(visibility_maps)
    visibility_maps = visibility_maps.reshape(-1, map_dims[2], map_dims[0], map_dims[1])
    visibility_maps = np.swapaxes(visibility_maps,2,3)  # annotate when generating mesh for coordinate issues - > car heading y
    visibility_maps = np.transpose(visibility_maps,(0,2,3,1))
    
    vis_occupy_indices = np.asarray(np.where(visibility_maps>0)).astype(np.uint8)
    vis_free_indices = np.asarray(np.where(visibility_maps<0)).astype(np.uint8)

    recover = np.zeros_like(visibility_maps, dtype = np.uint8) # for visualizations: uint16; for training: uint8
    recover[vis_occupy_indices[0,:],vis_occupy_indices[1,:],vis_occupy_indices[2,:],vis_occupy_indices[3,:]] = 1 #math.log(0.7/(1-0.7))
    recover[vis_free_indices[0,:],vis_free_indices[1,:],vis_free_indices[2,:],vis_free_indices[3,:]] = 0 #math.log(0.4/(1-0.4))


    # do packing 
    visibility_map_bin_pack = pack(recover)
    visibility_map_bin_pack.tofile(os.path.join(output_folder, os.path.splitext(f)[0] + ".pseudo"))

    # visibility_map_bin = np.array(recover.reshape(-1))
    # visibility_map_bin.tofile(os.path.join(output_folder, os.path.splitext(f)[0] + ".pseudo"))


    # print(float(os.path.splitext(f)[0]))
    # print("Finished processing:",float(os.path.splitext(f)[0]))


if __name__ == '__main__':
  start_time = time.time()

  parser = argparse.ArgumentParser("./lidar2voxel.py")
  parser.add_argument(
      '--dataset',
      '-d',
      type=str,
      required=True,
      help='dataset folder containing all sequences in a folder called "sequences".',
  )

  parser.add_argument(
      '--output',
      '-o',
      type=str,
      required=True,
      help='output folder for generated sequence scans.',
  )
  parser.add_argument(
      '--num_seq',
      '-n',
      type=str,
      required=True,
      help='number of sequence',
  )

  parser.add_argument(
      '--model',
      '-m',
      type=str,
      default='msnet3d',
      help='depth model',
  )

  parser.add_argument(
      '--sequence_length',
      '-s',
      type=int,
      default=1,
      help='length of sequence, i.e., how many scans are concatenated.',
  )

  
  FLAGS, unparsed = parser.parse_known_args()
  dataset = FLAGS.dataset
  output = FLAGS.output
  num_seq = FLAGS.num_seq

  print("*" * 80)
  print(" dataset folder: ", FLAGS.dataset)
  print("  output folder: ", FLAGS.output)
  print("sequence length: ", FLAGS.sequence_length)
  print("*" * 80)


  if FLAGS.sequence_length==1:
    sequences_dir = os.path.join(dataset, "sequences")
    input_folder = os.path.join(sequences_dir, num_seq)

    pseudo_lidar_files = [
          f for f in sorted(os.listdir(input_folder))
          if f.endswith(".bin")
      ]
    # output_folder = os.path.join(output, "sequences", num_seq, "voxels")
    # output_folder = os.path.join(output, "sequences_msnet3d_sweep1", num_seq, "voxels")
    # output_folder = os.path.join(output, "sequences_msnet3d_sweep1", num_seq, "voxels")
    output_folder = os.path.join(output, "sequences_" + FLAGS.model + "_sweep"+str(FLAGS.sequence_length), num_seq, "voxels")

    # mesh_temp_folder = os.path.join(output_folder, "temp")
    # mesh_folder = os.path.join(output_folder, "mesh")
    # scan_folder = os.path.join(output, "sequences", str(num_scan+1) + num_seq, "voxels")

    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    process_map(parallel_work_sequence, pseudo_lidar_files, max_workers=18)

    print("finished.")
    print("execution time: {}".format(time.time() - start_time))
  
  else:
    folder=str(num_seq)
    sequences_dir = os.path.join(FLAGS.dataset, "sequences")
    sequence_folders = [
        f for f in sorted(os.listdir(sequences_dir))
        if os.path.isdir(os.path.join(sequences_dir, f))
    ]

    # for folder in ["19"]:
    input_folder = os.path.join(sequences_dir, num_seq)
    output_folder = os.path.join(FLAGS.output, "sequences_" + FLAGS.model + "_sweep"+str(FLAGS.sequence_length), folder)

    scan_files = [
        f for f in sorted(os.listdir(os.path.join(input_folder)))
        if f.endswith(".bin")
    ]

    history = deque()

    calibration = parse_calibration(os.path.join(input_folder, "calib.txt"))
    poses = parse_poses(os.path.join(input_folder, "poses.txt"), calibration)

    progress = 10

    print("Processing {} ".format(folder), end="", flush=True)

    for i, f in enumerate(scan_files):
      # read scan and labels, get pose
      scan_filename = os.path.join(input_folder, f)
      scan = np.fromfile(scan_filename, dtype=np.float32)

      scan = scan.reshape((-1, 4))

      # label_filename = os.path.join(input_folder, "labels", os.path.splitext(f)[0] + ".label")
      # labels = np.fromfile(label_filename, dtype=np.uint32)
      # labels = labels.reshape((-1))

      # convert points to homogenous coordinates (x, y, z, 1)
      points = np.ones((scan.shape))
      points[:, 0:3] = scan[:, 0:3]
      remissions = scan[:, 3]

      pose = poses[i]

      # prepare single numpy array for all points that can be written at once.
      num_concat_points = points.shape[0]
      num_concat_points += sum([past["points"].shape[0] for past in history])
      concated_points = np.zeros((num_concat_points * 4), dtype = np.float32)
      # concated_labels = np.zeros((num_concat_points), dtype = np.uint32)

      start = 0
      concated_points[4 * start:4 * (start + points.shape[0])] = scan.reshape((-1))
      # concated_labels[start:start + points.shape[0]] = labels
      start += points.shape[0]

      for past in history:
        diff = np.matmul(inv(pose), past["pose"])
        tpoints = np.matmul(diff, past["points"].T).T
        tpoints[:, 3] = past["remissions"]
        tpoints = tpoints.reshape((-1))

        concated_points[4 * start:4 * (start + past["points"].shape[0])] = tpoints
        # concated_labels[start:start + past["labels"].shape[0]] = past["labels"]
        start += past["points"].shape[0]

      if float(os.path.splitext(f)[0])%5==0:
        print(float(os.path.splitext(f)[0]))
      # if True:
        #------------------------------------------------------------------------------------------------------------------------  
        #------------------------------------------------------------------------------------------------------------------------  
        # (1). visibility (voxel) generation
        visibility_maps = []
        # origins = np.zeros((1,4))
        origins = np.array([[0, 0, 0, 1]])
        map_dims = [256, 256, 32]
        voxel_size = (0.2, 0.2, 0.2)
        area_extents = np.array([[0, 51.2], [-25.6, 25.6], [-2., 4.4]])
        
        pc_range = [area_extents[0,0], area_extents[1,0], area_extents[2,0], area_extents[0,1], area_extents[1,1], area_extents[2,1]]
        pts = concated_points.reshape(-1,4)
        filter_idx = np.where((area_extents[0, 0] < pts[:, 0]) & (pts[:, 0] < area_extents[0, 1]) & (area_extents[1, 0] < pts[:, 1]) & (pts[:, 1] < area_extents[1, 1]) & (area_extents[2, 0] < pts[:, 2]) & (pts[:, 2] < area_extents[2, 1]))[0]
        pts = pts[filter_idx]
          
        visibility_maps.append(mapping.compute_logodds_dp(pts, origins[[0],:3], pc_range, range(pts.shape[0]), 0.2)) #, lo_occupied, lo_free
        visibility_maps = np.asarray(visibility_maps)
        visibility_maps = visibility_maps.reshape(-1, map_dims[2], map_dims[0], map_dims[1])
        visibility_maps = np.swapaxes(visibility_maps,2,3)  # annotate when generating mesh for coordinate issues - > car heading y
        visibility_maps = np.transpose(visibility_maps,(0,2,3,1))
        
        vis_occupy_indices = np.asarray(np.where(visibility_maps>0)).astype(np.uint8)
        vis_free_indices = np.asarray(np.where(visibility_maps<0)).astype(np.uint8)

        recover = np.zeros_like(visibility_maps, dtype = np.uint8) # for visualizations: uint16; for training: uint8
        recover[vis_occupy_indices[0,:],vis_occupy_indices[1,:],vis_occupy_indices[2,:],vis_occupy_indices[3,:]] = 1 #math.log(0.7/(1-0.7))
        recover[vis_free_indices[0,:],vis_free_indices[1,:],vis_free_indices[2,:],vis_free_indices[3,:]] = 0 #math.log(0.4/(1-0.4))

        # visibility_map_bin = np.array(recover.reshape(-1))
        visibility_map_bin = pack(recover)

        voxel_output_folder = os.path.join(output_folder, "voxels")
        if not os.path.exists(voxel_output_folder):
          os.makedirs(voxel_output_folder)

        visibility_map_bin.tofile(os.path.join(voxel_output_folder, os.path.splitext(f)[0] + ".pseudo"))
        # print(float(os.path.splitext(f)[0]))

        # print("Finished processing:",float(os.path.splitext(f)[0]))

      # append current data to history queue.
      history.appendleft({
          "points": points,
          "remissions": remissions,
          "pose": pose.copy()
      })

      if len(history) >= FLAGS.sequence_length:
        history.pop()

      if 100.0 * i / len(scan_files) >= progress:
        print(".", end="", flush=True)
        progress = progress + 10
    print("finished.")


  print("execution time: {}".format(time.time() - start_time))
