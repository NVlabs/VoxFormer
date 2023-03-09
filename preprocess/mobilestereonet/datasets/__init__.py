from .dataset import SceneFlowDataset, KITTIDataset, KITTI360Dataset, DrivingStereoDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "kitti360": KITTI360Dataset,
    "drivingstereo": DrivingStereoDataset,
}
