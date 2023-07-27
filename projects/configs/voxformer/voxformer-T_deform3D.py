work_dir = 'result/voxformer-T_deform3D'
_base_ = [
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_layers_self_ = 2
_num_points_self_ = 8
_dim_ = 128
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1

_labels_tag_ = 'labels'
_num_cams_ = 5
_temporal_ = [-12,-9,-6,-3]
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]

_sem_scal_loss_ = True
_geo_scal_loss_ = True
_depthmodel_= 'msnet3d'
_nsweep_ = 10
_query_tag_ = 'query_iou5203_pre7712_rec6153'

model = dict(
   type='VoxFormer',
   pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
   img_backbone=dict(
       type='ResNet',
       depth=50,
       num_stages=4,
       out_indices=(2,),
       frozen_stages=1,
       norm_cfg=dict(type='BN', requires_grad=False),
       norm_eval=True,
       style='pytorch'),
   img_neck=dict(
       type='FPN',
       in_channels=[1024],
       out_channels=_dim_,
       start_level=0,
       add_extra_convs='on_output',
       num_outs=_num_levels_,
       relu_before_extra_convs=True),
   pts_bbox_head=dict(
       type='VoxFormerHead',
       bev_h=128,
       bev_w=128,
       bev_z=16,
       embed_dims=_dim_,
       CE_ssc_loss=True,
       geo_scal_loss=_geo_scal_loss_,
       sem_scal_loss=_sem_scal_loss_,
       cross_transformer=dict(
           type='PerceptionTransformer',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='VoxFormerEncoder',
               num_layers=_num_layers_cross_,
               pc_range=point_cloud_range,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformCrossAttention',
                           pc_range=point_cloud_range,
                           num_cams=_num_cams_,
                           deformable_attention=dict(
                               type='MSDeformableAttention3D',
                               embed_dims=_dim_,
                               num_points=_num_points_cross_,
                               num_levels=_num_levels_),
                           embed_dims=_dim_,
                       )
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_ffn_dim_,
                   ffn_dropout=0.1,
                   operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
       self_transformer=dict(
           type='PerceptionTransformer3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='VoxFormerEncoder3D',
               num_layers=_num_layers_self_,
               pc_range=point_cloud_range,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer3D',
                   attn_cfgs=[
                       dict(
                           type='DeformSelfAttention3DCustom',
                           embed_dims=_dim_,
                           num_levels=1,
                           num_points=_num_points_self_)
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_ffn_dim_,
                   ffn_dropout=0.1,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
       positional_encoding=dict(
           type='LearnedPositionalEncoding',
           num_feats=_pos_dim_,
           row_num_embed=512,
           col_num_embed=512,
           )),
   train_cfg=dict(pts=dict(
       grid_size=[512, 512, 1],
       voxel_size=voxel_size,
       point_cloud_range=point_cloud_range,
       out_size_factor=4)))


dataset_type = 'SemanticKittiDatasetStage2'
data_root = './kitti/'
file_client_args = dict(backend='disk')

data = dict(
   samples_per_gpu=1,
   workers_per_gpu=4,
   train=dict(
       type=dataset_type,
       split = "train",
       test_mode=False,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   val=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   test=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   shuffler_sampler=dict(type='DistributedGroupSampler'),
   nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
   type='AdamW',
   lr=2e-4,
   weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
   policy='CosineAnnealing',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=1.0 / 3,
   min_lr_ratio=1e-3)
total_epochs = 20
evaluation = dict(interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
   interval=50,
   hooks=[
       dict(type='TextLoggerHook'),
       dict(type='TensorboardLoggerHook')
   ])

checkpoint_config = None
# checkpoint_config = dict(interval=2)
