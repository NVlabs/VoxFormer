/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
**************************************************************************************************
*/
#ifndef DEFORM_ATTN_CUDA_KERNEL
#define DEFORM_ATTN_CUDA_KERNEL

#include "common_cuda_helper.hpp"
#include "pytorch_cuda_helper.hpp"

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,const int &depth,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const scalar_t &z, const int &m, const int &c) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int z_low = floorf(z);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const int z_high = z_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t lz = z - z_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw, hz = 1 - lz;

  const int z_stride = nheads * channels;
  const int w_stride = depth * z_stride;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int z_low_ptr_offset = z_low * z_stride;
  const int z_high_ptr_offset = z_low_ptr_offset + z_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0 && z_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + z_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1 && z_low >= 0) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + z_low_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0 && z_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + z_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && z_low >= 0) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + z_low_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }
  scalar_t v5 = 0;
  if (h_low >= 0 && w_low >= 0 && z_high <= depth - 1) {
    const int ptr5 = h_low_ptr_offset + w_low_ptr_offset + z_high_ptr_offset  + base_ptr;
    v5 = bottom_data[ptr5];
  }
  scalar_t v6 = 0;
  if (h_low >= 0 && w_high <= width - 1 && z_high <= depth - 1) {
    const int ptr6 = h_low_ptr_offset + w_high_ptr_offset + z_high_ptr_offset  + base_ptr;
    v6 = bottom_data[ptr6];
  }
  scalar_t v7 = 0;
  if (h_high <= height - 1 && w_low >= 0 && z_high <= depth - 1) {
    const int ptr7 = h_high_ptr_offset + w_low_ptr_offset + z_high_ptr_offset  + base_ptr;
    v7 = bottom_data[ptr7];
  }
  scalar_t v8 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && z_high <= depth - 1) {
    const int ptr8 = h_high_ptr_offset + w_high_ptr_offset + z_high_ptr_offset  + base_ptr;
    v8 = bottom_data[ptr8];
  }

  const scalar_t w1 = hh * hw * hz, w2 = hh * lw * hz, w3 = lh * hw * hz, w4 = lh * lw * hz;
  const scalar_t w5 = hh * hw * lz, w6 = hh * lw * lz, w7 = lh * hw * lz, w8 = lh * lw * lz;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  return val;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,const int &depth,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w,const scalar_t &z, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int z_low = floorf(z);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const int z_high = z_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t lz = z - z_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw, hz = 1 - lz;

  const int z_stride = nheads * channels;
  const int w_stride = depth * z_stride;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int z_low_ptr_offset = z_low * z_stride;
  const int z_high_ptr_offset = z_low_ptr_offset + z_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw * hz, w2 = hh * lw * hz, w3 = lh * hw * hz, w4 = lh * lw * hz;
  const scalar_t w5 = hh * hw * lz, w6 = hh * lw * lz, w7 = lh * hw * lz, w8 = lh * lw * lz;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0, grad_z_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0 && z_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + z_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    grad_z_weight -= hz * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1 && z_low >= 0) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + z_low_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    grad_z_weight -= hz * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0 && z_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + z_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    grad_z_weight -= hz * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && z_low >= 0) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + z_low_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    grad_z_weight -= hz * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }
  scalar_t v5 = 0;
  if (h_low >= 0 && w_low >= 0 && z_high <= depth - 1) {
    const int ptr5 = h_low_ptr_offset + w_low_ptr_offset + z_high_ptr_offset + base_ptr;
    v5 = bottom_data[ptr5];
    grad_h_weight -= hw * v5;
    grad_w_weight -= hh * v5;
    grad_z_weight += lz * v5;
    atomicAdd(grad_value + ptr5, w5 * top_grad_value);
  }
  scalar_t v6 = 0;
  if (h_low >= 0 && w_high <= width - 1 && z_high <= depth - 1) {
    const int ptr6 = h_low_ptr_offset + w_high_ptr_offset + z_high_ptr_offset + base_ptr;
    v6 = bottom_data[ptr6];
    grad_h_weight -= lw * v6;
    grad_w_weight += hh * v6;
    grad_z_weight += lz * v6;
    atomicAdd(grad_value + ptr6, w6 * top_grad_value);
  }
  scalar_t v7 = 0;
  if (h_high <= height - 1 && w_low >= 0 && z_high <= depth - 1) {
    const int ptr7 = h_high_ptr_offset + w_low_ptr_offset + z_high_ptr_offset + base_ptr;
    v7 = bottom_data[ptr7];
    grad_h_weight += hw * v7;
    grad_w_weight -= lh * v7;
    grad_z_weight += lz * v7;
    atomicAdd(grad_value + ptr7, w7 * top_grad_value);
  }
  scalar_t v8 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && z_high <= depth - 1) {
    const int ptr8 = h_high_ptr_offset + w_high_ptr_offset + z_high_ptr_offset + base_ptr;
    v8 = bottom_data[ptr8];
    grad_h_weight += lw * v8;
    grad_w_weight += lh * v8;
    grad_z_weight += lz * v8;
    atomicAdd(grad_value + ptr8, w8 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
  *(grad_sampling_loc + 2) = depth * grad_z_weight * top_grad_value;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(
    const scalar_t *&bottom_data, const int &height, const int &width,const int &depth,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w,const scalar_t &z, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int z_low = floorf(z);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const int z_high = z_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t lz = z - z_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw, hz = 1 - lz;

  const int z_stride = nheads * channels;
  const int w_stride = depth * z_stride;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int z_low_ptr_offset = z_low * z_stride;
  const int z_high_ptr_offset = z_low_ptr_offset + z_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw * hz, w2 = hh * lw * hz, w3 = lh * hw * hz, w4 = lh * lw * hz;
  const scalar_t w5 = hh * hw * lz, w6 = hh * lw * lz, w7 = lh * hw * lz, w8 = lh * lw * lz;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0, grad_z_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0 && z_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + z_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    grad_z_weight -= hz * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1 && z_low >= 0) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + z_low_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    grad_z_weight -= hz * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0 && z_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + z_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    grad_z_weight -= hz * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && z_low >= 0) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + z_low_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    grad_z_weight -= hz * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }
  scalar_t v5 = 0;
  if (h_low >= 0 && w_low >= 0 && z_high <= depth - 1) {
    const int ptr5 = h_low_ptr_offset + w_low_ptr_offset + z_high_ptr_offset + base_ptr;
    v5 = bottom_data[ptr5];
    grad_h_weight -= hw * v5;
    grad_w_weight -= hh * v5;
    grad_z_weight += lz * v5;
    atomicAdd(grad_value + ptr5, w5 * top_grad_value);
  }
  scalar_t v6 = 0;
  if (h_low >= 0 && w_high <= width - 1 && z_high <= depth - 1) {
    const int ptr6 = h_low_ptr_offset + w_high_ptr_offset + z_high_ptr_offset + base_ptr;
    v6 = bottom_data[ptr6];
    grad_h_weight -= lw * v6;
    grad_w_weight += hh * v6;
    grad_z_weight += lz * v6;
    atomicAdd(grad_value + ptr6, w6 * top_grad_value);
  }
  scalar_t v7 = 0;
  if (h_high <= height - 1 && w_low >= 0 && z_high <= depth - 1) {
    const int ptr7 = h_high_ptr_offset + w_low_ptr_offset + z_high_ptr_offset + base_ptr;
    v7 = bottom_data[ptr7];
    grad_h_weight += hw * v7;
    grad_w_weight -= lh * v7;
    grad_z_weight += lz * v7;
    atomicAdd(grad_value + ptr7, w7 * top_grad_value);
  }
  scalar_t v8 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && z_high <= depth - 1) {
    const int ptr8 = h_high_ptr_offset + w_high_ptr_offset + z_high_ptr_offset + base_ptr;
    v8 = bottom_data[ptr8];
    grad_h_weight += lw * v8;
    grad_w_weight += lh * v8;
    grad_z_weight += lz * v8;
    atomicAdd(grad_value + ptr8, w8 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  atomicAdd(grad_attn_weight, top_grad * val);
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 2, depth * grad_z_weight * top_grad_value);
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    scalar_t *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col *3;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_z = data_spatial_shapes[spatial_h_ptr + 2];
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_z = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t z_im = loc_z * spatial_z - 0.5;

        if (h_im > -1 && w_im > -1 && z_im > -1 
        && h_im < spatial_h && w_im < spatial_w && z_im < spatial_z) {
          col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h,
                                                spatial_w,spatial_z, num_heads, channels,
                                                h_im, w_im,z_im, m_col, c_col) *
                 weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  __shared__ scalar_t cache_grad_sampling_loc[blockSize * 3];
  __shared__ scalar_t cache_grad_attn_weight[blockSize];
  unsigned int tid = threadIdx.x;
  const int qid_stride = num_heads * channels;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr*3;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr*3);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col *3;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_z = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_z = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t z_im = loc_z * spatial_z - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x *3)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 2)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && z_im > -1
        && h_im < spatial_h 
        && w_im < spatial_w
        && z_im < spatial_z
        ) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, spatial_z, num_heads, channels, h_im,
              w_im,z_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x *3),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_z = cache_grad_sampling_loc[2],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 3;
          for (unsigned int _tid = 1; _tid < blockSize; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_z += cache_grad_sampling_loc[sid + 2];
            _grad_a += cache_grad_attn_weight[_tid];
            sid += 3;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *(grad_sampling_loc_out + 2) = _grad_z;
          *grad_attn_weight_out = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
  __shared__ scalar_t cache_grad_attn_weight[blockSize];
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr *3;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr *3);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col *3;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_z = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_z = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t z_im = loc_z * spatial_z - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x *3)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 2)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && z_im > -1
        && h_im < spatial_h 
        && w_im < spatial_w
        && z_im < spatial_z
        ) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w,spatial_z, num_heads, channels, h_im,
              w_im,z_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x *3),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockSize / 3; s > 0; s *= 3) {
          if (tid < s) {
            const unsigned int xid1 = tid *3;
            const unsigned int xid2 = (tid + s) *3;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            cache_grad_sampling_loc[xid1 + 2] +=
                cache_grad_sampling_loc[xid2 + 2];
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc_out = cache_grad_sampling_loc[0];
          *(grad_sampling_loc_out + 1) = cache_grad_sampling_loc[1];
          *(grad_sampling_loc_out + 2) = cache_grad_sampling_loc[2];
          *grad_attn_weight_out = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 3 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr *3;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr *3);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col *3;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_z = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_z = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t z_im = loc_z * spatial_z - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x *3)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 2)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && z_im > -1
        && h_im < spatial_h 
        && w_im < spatial_w
        && z_im < spatial_z
        ) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, spatial_z, num_heads, channels, h_im,
              w_im, z_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x *3),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_z = cache_grad_sampling_loc[2],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 3;
          for (unsigned int _tid = 1; _tid < blockDim.x; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_z += cache_grad_sampling_loc[sid + 2];
            _grad_a += cache_grad_attn_weight[_tid];
            sid += 3;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *(grad_sampling_loc_out + 2) = _grad_z;
          *grad_attn_weight_out = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 3 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr *3;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr *3);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col *3;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_z = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_z = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t z_im = loc_z * spatial_z - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x *3)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 2)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && z_im > -1
        && h_im < spatial_h 
        && w_im < spatial_w
        && z_im < spatial_z
        ) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, spatial_z, num_heads, channels, h_im,
              w_im, z_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x *3),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 3, spre = blockDim.x; s > 0;
             s *=3, spre *=3) {
          if (tid < s) {
            const unsigned int xid1 = tid *3;
            const unsigned int xid2 = (tid + s) *3;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            cache_grad_sampling_loc[xid1 + 2] +=
                cache_grad_sampling_loc[xid2 + 2];
            if (tid + (s *3) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s *3)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s *3)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s *3)];
              cache_grad_sampling_loc[xid1 + 2] +=
                  cache_grad_sampling_loc[xid2 + 2 + (s *3)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc_out = cache_grad_sampling_loc[0];
          *(grad_sampling_loc_out + 1) = cache_grad_sampling_loc[1];
          *(grad_sampling_loc_out + 2) = cache_grad_sampling_loc[2];
          *grad_attn_weight_out = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 3 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr *3;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr *3);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_z = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_z = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t z_im = loc_z * spatial_z - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x *3)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x *3) + 2)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && z_im > -1
        && h_im < spatial_h 
        && w_im < spatial_w
        && z_im < spatial_z
        ) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, spatial_z, num_heads, channels, h_im,
              w_im, z_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x *3),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 3, spre = blockDim.x; s > 0;
             s *= 3, spre *=3) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            cache_grad_sampling_loc[xid1 + 2] +=
                cache_grad_sampling_loc[xid2 + 2];
            if (tid + (s *3) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s *3)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s *3)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s *3)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          atomicAdd(grad_sampling_loc_out, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc_out + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_sampling_loc_out + 2, cache_grad_sampling_loc[2]);
          atomicAdd(grad_attn_weight_out, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col *3;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_z = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
   const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_z = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t z_im = loc_z * spatial_z - 0.5;

        if (h_im > -1 && w_im > -1 && z_im > -1
        && h_im < spatial_h 
        && w_im < spatial_w
        && z_im < spatial_z
        ) {
          ms_deform_attn_col2im_bilinear_gm(
              data_value_ptr, spatial_h, spatial_w, spatial_z, num_heads, channels, h_im,
              w_im, z_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              grad_sampling_loc_out, grad_attn_weight_out);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}
#endif  // DEFORM_ATTN_CUDA_KERNEL
