from __future__ import print_function
import torch
import numpy as np
import torch.utils.data
# from torch.autograd import Function


def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


error_colormap = gen_error_colormap()


# class disp_error_image_func(Function):
#     def forward(self, D_est_tensor, D_gt_tensor, abs_thres=3., rel_thres=0.05, dilate_radius=1):
#         D_gt_np = D_gt_tensor.detach().cpu().numpy()
#         D_est_np = D_est_tensor.detach().cpu().numpy()
#         B, H, W = D_gt_np.shape
#         # valid mask
#         mask = D_gt_np > 0
#         # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
#         error = np.abs(D_gt_np - D_est_np)
#         error[np.logical_not(mask)] = 0
#         error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)
#         # get colormap
#         cols = error_colormap
#         # create error image
#         error_image = np.zeros([B, H, W, 3], dtype=np.float32)
#         for i in range(cols.shape[0]):
#             error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
#         error_image[np.logical_not(mask)] = 0.
#         # show color tag in the top-left corner of the image
#         for i in range(cols.shape[0]):
#             distance = 20
#             error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
#
#         return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))
#
#     def backward(self, grad_output):
#         return None


def disp_error_image_func(D_est_tensor, D_gt_tensor, abs_thres=3., rel_thres=0.05, dilate_radius=1):

    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    B, H, W = D_gt_np.shape

    # valid mask
    mask = D_gt_np > 0

    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)

    # get colormap
    cols = error_colormap

    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    error_image[np.logical_not(mask)] = 0.

    # show color tag in the top-left corner of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))

