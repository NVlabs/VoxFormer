

## SemanticKITTI
### Two options

1. Direct downloading

      - The **semantic scene completion dataset v1.1** (SemanticKITTI voxel data, 700 MB) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download).
      -  The **RGB images** (Download odometry data set (color, 65 GB)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
      -  The **calibration and pose** files from voxformer/preprocess/data_odometry_calib/sequences.
      -  The **preprocessed ground truth** (~700MB) from [labels](https://drive.google.com/file/d/1r6RWjPClt9-EBbuOczLB295c00o7pOOP/view?usp=share_link).
      -  The **voxelized psuedo point cloud** and **query proposals** (~400MB) based on MobileStereoNet from [sequences_msnet3d_sweep10](https://drive.google.com/file/d/1nxWC3z4D4LDboQoMA-mnlJ7QHUnR9gRn/view?usp=share_link).

2. Downloading the voxel and image data first, then following the commands in *voxformer/preprocess* to create **labels** and **sequences_msnet3d_sweep10**. You need to choose this option if you would like to use different data or depth models.

### Folder structure

The data is organized in the following format:

```
/kitti/dataset/
          └── sequences/
          │       ├── 00/
          │       │   ├── poses.txt
          │       │   ├── calib.txt
          │       │   ├── image_2/
          │       │   ├── image_3/
          │       |   ├── voxels/
          │       |         ├ 000000.bin
          │       |         ├ 000000.label
          │       |         ├ 000000.occluded
          │       |         ├ 000000.invalid
          │       |         ├ 000005.bin
          │       |         ├ 000005.label
          │       |         ├ 000005.occluded
          │       |         ├ 000005.invalid
          │       ├── 01/
          │       ├── 02/
          │       .
          │       └── 21/
          └── labels/
          │       ├── 00/
          │       │   ├── 000000_1_1.npy
          │       │   ├── 000000_1_2.npy
          │       │   ├── 000005_1_1.npy
          │       │   ├── 000005_1_2.npy
          │       ├── 01/
          │       .
          │       └── 10/
          └── sequences_msnet3d_sweep10/
                  ├── 00/
                  │   ├── voxels/
                  │   │     ├ 000000.pseudo
                  │   │     ├ 000005.pseudo
                  │   ├── queries/
                  │   │     ├ 000000.query
                  │   │     ├ 000005.query
                  ├── 01/
                  ├── 02/
                  .
                  └── 21/
```
