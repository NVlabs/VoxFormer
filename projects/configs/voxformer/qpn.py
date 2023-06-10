_gamma_=0
_alpha_=0.54
_nsweep_=10
_depthmodel_ = "msnet3d"

_base_ = [
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 128
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 128
bev_w_ = 128
queue_length = 3 # each sequence contains `queue_length` frames.

model = dict(
    type='LMSCNet_SS',
    class_num=2, 
    input_dimensions=[256, 32, 256],
    out_scale = "1_2",
    gamma = _gamma_,
    alpha = _alpha_,
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'SemanticKittiDatasetStage1'
data_root = './kitti/'
file_client_args = dict(backend='disk')

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True)
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        split = "train",
        test_mode=False,
        data_root=data_root,
        preprocess_root=data_root + 'dataset',
        nsweep=_nsweep_,
        depthmodel=_depthmodel_),
    val=dict(
        type=dataset_type,
        split = "val",
        test_mode=True,
        data_root=data_root,
        preprocess_root=data_root + 'dataset',
        nsweep=_nsweep_,
        depthmodel=_depthmodel_),
    test=dict(
        type=dataset_type,
        split = "val",
        test_mode=True,
        data_root=data_root,
        preprocess_root=data_root + 'dataset',
        nsweep=_nsweep_,
        depthmodel=_depthmodel_),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'img_backbone': dict(lr_mult=0.1),
    #     }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# checkpoint_config = None
checkpoint_config = dict(interval=1)
