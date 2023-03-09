#
# MIT License
#
# Copyright (c) 2019 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import cv2


def kitti_colormap(disparity, maxval=-1):
    """
	A utility function to reproduce KITTI fake colormap
	Arguments:
	  - disparity: numpy float32 array of dimension HxW
	  - maxval: maximum disparity value for normalization (if equal to -1, the maximum value in disparity will be used)
	
	Returns a numpy uint8 array of shape HxWx3.
	"""
    if maxval < 0:
        maxval = np.max(disparity)

    colormap = np.asarray(
        [[0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174], [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114],
         [1, 1, 1, 0]])
    weights = np.asarray([8.771929824561404, 5.405405405405405, 8.771929824561404, 5.747126436781609, 8.771929824561404,
                          5.405405405405405, 8.771929824561404, 0])
    cumsum = np.asarray([0, 0.114, 0.299, 0.413, 0.587, 0.701, 0.8859999999999999, 0.9999999999999999])

    colored_disp = np.zeros([disparity.shape[0], disparity.shape[1], 3])
    values = np.expand_dims(np.minimum(np.maximum(disparity / maxval, 0.), 1.), -1)
    bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum, axis=0), axis=0), disparity.shape[1], axis=1),
                     disparity.shape[0], axis=0)
    diffs = np.where((np.repeat(values, 8, axis=-1) - bins) > 0, -1000, (np.repeat(values, 8, axis=-1) - bins))
    index = np.argmax(diffs, axis=-1) - 1

    w = 1 - (values[:, :, 0] - cumsum[index]) * np.asarray(weights)[index]

    colored_disp[:, :, 2] = (w * colormap[index][:, :, 0] + (1. - w) * colormap[index + 1][:, :, 0])
    colored_disp[:, :, 1] = (w * colormap[index][:, :, 1] + (1. - w) * colormap[index + 1][:, :, 1])
    colored_disp[:, :, 0] = (w * colormap[index][:, :, 2] + (1. - w) * colormap[index + 1][:, :, 2])

    return (colored_disp * np.expand_dims((disparity > 0), -1) * 255).astype(np.uint8)


def read_16bit_gt(path):
    """
	A utility function to read KITTI 16bit gt
	Arguments:
	  - path: filepath 	
	Returns a numpy float32 array of shape HxW.
	"""
    gt = cv2.imread(path, -1).astype(np.float32) / 256.
    return gt
