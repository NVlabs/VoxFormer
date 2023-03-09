## 1. Prepare data
Symlink the dataset root to ./kitti.
```
ln -s [SemanticKITTI root] ./kitti
```
Copy the calibration files to dataset
```
cp -r data_odometry_calib/sequences kitti/dataset
```
The data is organized in the following format:

```
/kitti/dataset/
          └── sequences/
                  ├── 00/
                  │   ├── poses.txt
                  │   ├── calib.txt
                  │   ├── image_2/
                  │   ├── image_3/
                  |   ├── voxels/
                  |         ├ 000000.bin
                  |         ├ 000000.label
                  |         ├ 000000.occluded
                  |         ├ 000000.invalid
                  |         ├ 000005.bin
                  |         ├ 000005.label
                  |         ├ 000005.occluded
                  |         ├ 000005.invalid
                  ├── 01/
                  ├── 02/
                  .
                  └── 21/

```
## 2. Generating grounding truth
Setting up the environment
```shell
conda create -n preprocess python=3.7 -y
conda activate preprocess
conda install numpy tqdm pyyaml imageio
```
Preprocess the data to generate labels at a lower scale:
```
python label/label_preprocess.py --kitti_root=[SemanticKITTI root] --kitti_preprocess_root=[preprocess_root]
```

Then we have the following data:
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
                  ├── 00/
                  │   ├── 000000_1_1.npy
                  │   ├── 000000_1_2.npy
                  │   ├── 000005_1_1.npy
                  │   ├── 000005_1_2.npy
                  ├── 01/
                  .
                  └── 10/

```

## 3. Image to depth
### Disparity estimation
We use [MobileStereoNet3d](https://github.com/cogsys-tuebingen/mobilestereonet) to obtain the disparity. We add several lines to convert disparity into depth, and add filenames to support kitti odometry dataset. We upload our folder for your convenience. Please refer to the [original repository](https://github.com/cogsys-tuebingen/mobilestereonet) for detailed instructions.

### Requirements
The code is tested on:
- Ubuntu 18.04
- Python 3.6 
- PyTorch 1.4.0 
- Torchvision 0.5.0
- CUDA 10.0

### Setting up the environment

```shell
cd mobilestereonet
conda env create --file mobilestereonet.yaml # please modify prefix in .yaml
conda activate mobilestereonet
```

### Prediction

The following script could create depth maps for all sequences:
```shell
./image2depth.sh
```
## 4. Depth to pseudo point cloud
The following script could create pseudo point cloud for all sequences:

```shell
./depth2lidar.sh
```
## 5. Point cloud to voxel
### Setting up the environment

```shell
conda activate preprocess
```

### (Optional) Build from source
```shell
cd mapping 
# delete cmake cache file
# set the path to eigen in CMakeLists.txt
cmake ./ 
make all
```

### sweep voxelization
```shell
./lidar2voxel.sh
```
The input of the **query proposal network** will be created in *./kitti/dataset/sequences_msnet3d_sweep[sequence_length]*.

Finally we have the following data:
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
