# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
import glob
import copy
import torch
import random
import mmcv
import numpy as np
from torch.utils.data import Dataset
from os import path as osp
from PIL import Image
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.voxformer.utils.ssc_metric import SSCMetrics

@DATASETS.register_module()
class SemanticKittiDatasetStage1(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        depthmodel="sdnnet",
        nsweep=5,
    ):
        super().__init__()
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, "labels")
        self.n_classes = 2
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }

        self.nsweep=str(nsweep)
        self.depthmodel = depthmodel
        self.class_names =  [ "empty", "occupied"]
        self.metrics = SSCMetrics(2)
        self.split = split
        self.sequences = splits[split]

        self.voxel_size = 0.2  # 0.2m
        self.img_W = 1220
        self.img_H = 370
        
        self.load_scans()
        self.test_mode = test_mode
        self._set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    def load_scans(self):
        self.scans = []
        for sequence in self.sequences:

            glob_path = os.path.join(
                self.data_root, "dataset", "sequences_" + self.depthmodel + "_sweep"+ self.nsweep, sequence, "voxels", "*.pseudo"
            ) # os.path.join(self.data_root, "dataset", "sequences", sequence, "voxels", frame_id + ".bin")
            for voxel_path in glob.glob(glob_path):

                self.scans.append(
                    {
                        "sequence": sequence,
                        "voxel_path": voxel_path
                    }
                )

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []
        example = self.get_data_info(index)

        data_queue.insert(0, example)

        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'] for each in queue]
        metas_map = {}
        
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)

        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. 
        """
        scan = self.scans[index]
        voxel_path = scan["voxel_path"]
        sequence = scan["sequence"]
        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )
        image_paths = []
        image_paths.append(rgb_path)

        # load voxelized pseudo point cloud
        pseudo_pc_bin = self.read_occupancy_SemKITTI(voxel_path)

        if self.split == "train" or self.split == "val":
            # load ground truth
            target_1_2_path = os.path.join(self.label_root, sequence, frame_id + "_1_2.npy")
            target = np.load(target_1_2_path)
            target = target.reshape(-1)
            target = target.reshape(128, 128, 16)
            target = target.astype(np.float32)
        else:
            target = np.ones((128,128,16))

        meta_dict = dict(
            target=target,
            sequence_id = sequence,
            voxel=None,
            pseudo_pc=pseudo_pc_bin,
            img_filename=image_paths,
            img_shape = [(370,1220)]
        )

        data_info = dict(
            img_metas = meta_dict,
            img = torch.zeros((1, 3, 370, 1220)),
            target = meta_dict['target']
        )
 
        return data_info

    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed

    def pack(self, array):
        """ convert a boolean array into a bitwise array. """
        array = array.reshape((-1))

        #compressing bit flags.
        # yapf: disable
        compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
        # yapf: enable

        return np.array(compressed, dtype=np.uint8)

    def read_SemKITTI(self, path, dtype, do_unpack):
        bin = np.fromfile(path, dtype=dtype)  # Flattened array
        if do_unpack:
            bin = self.unpack(bin)
        return bin

    def read_occupancy_SemKITTI(self, path):
        occupancy = self.read_SemKITTI(path, dtype=np.uint8, do_unpack=True).astype(np.float32)
        return occupancy

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        detail = dict()

        for result in results:
            self.metrics.add_batch(result['y_pred'], result['y_true'])
        metric_prefix = f'{result_name}_SemanticKITTI'

        stats = self.metrics.get_stats()
        for i, class_name in enumerate(self.class_names):
            detail["{}/SemIoU_{}".format(metric_prefix, class_name)] = stats["iou_ssc"][i]

        detail["{}/mIoU".format(metric_prefix)] = stats["iou_ssc_mean"]
        detail["{}/IoU".format(metric_prefix)] = stats["iou"]
        detail["{}/Precision".format(metric_prefix)] = stats["precision"]
        detail["{}/Recall".format(metric_prefix)] = stats["recall"]

        self.metrics.reset()

        return detail
