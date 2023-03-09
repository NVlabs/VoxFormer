# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .deformable_self_attention import DeformSelfAttention
from .deformable_cross_attention import MSDeformableAttention3D

@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, DeformSelfAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_vox_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            ref_3d,
            vox_coords,
            unmasked_idx,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain voxel features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) #  #[N, 1, 64]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # [N, 1, 64]

        unmasked_bev_queries = bev_queries[vox_coords[unmasked_idx[0], 3], :, :]
        unmasked_bev_bev_pos = bev_pos[vox_coords[unmasked_idx[0], 3], :, :]

        unmasked_ref_3d = torch.from_numpy(ref_3d[vox_coords[unmasked_idx[0], 3], :]) 
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(unmasked_bev_queries.device)
        
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            unmasked_bev_queries,
            feat_flatten,
            feat_flatten,
            ref_3d=unmasked_ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=unmasked_bev_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=None,
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def diffuse_vox_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            ref_3d,
            vox_coords,
            unmasked_idx,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        diffuse voxel features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) 
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        unmasked_ref_3d = torch.from_numpy(ref_3d[vox_coords[unmasked_idx[0], 3], :]) 
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(bev_queries.device)
        
        bev_embed = self.encoder(
            bev_queries,
            None,
            None,
            ref_3d=unmasked_ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=None,
            level_start_index=None,
            prev_bev=None,
            shift=None,
            **kwargs
        ) 
        
        return bev_embed
