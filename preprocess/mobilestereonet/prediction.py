'''
Mathmatical Formula to convert the disparity to depth:

depth = baseline * focal / disparity
For KITTI the baseline is 0.54m and the focal ~721 pixels.
The final formula is:
depth = 0.54 * 721 / disp

For KITTI-360, depth = 0.6 * 552.554261 / disp
'''

from __future__ import print_function, division
import os
import argparse
import torch.nn as nn
from skimage import io
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import __datasets__
from models import __models__
from utils import *
from utils.KittiColormap import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='MobileStereoNet')
parser.add_argument('--model', default='MSNet2D', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--colored', default=1, help='save colored or save for benchmark submission')
parser.add_argument('--num_seq', type=str, default=00, help='number of sequence')
parser.add_argument('--savepath', required=True, help='save path')
parser.add_argument('--baseline', type=float, default=388.1823, help='baseline*focal')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("Loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test(args):
    print("Generating the disparity maps...")

    os.makedirs('./predictions', exist_ok=True)

    for batch_idx, sample in enumerate(TestImgLoader):

        disp_est_tn = test_sample(sample)
        disp_est_np = tensor2numpy(disp_est_tn)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):

            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32) 

            # -------------------------------------------------------------------------------------------------------------
            # convert to depth value
            output_folder = os.path.join(args.savepath, "sequences", args.num_seq)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            fn = os.path.join(output_folder, fn.split('/')[-1].split('.')[0])
            depth = args.baseline / disp_est.clip(min=1e-8)
            np.save(fn, depth)

            # depth = 388.1823 / disp_est.clip(min=1e-8) # sequence 0-2; 13-21
            # depth = 381.8293 / disp_est.clip(min=1e-8) # sequence 4-12
            # depth = 389.6304 / disp_est.clip(min=1e-8) # sequence 3
            # depth = 331.532557 / disp_est.clip(min=1e-8) # kitti-360
        
            # -------------------------------------------------------------------------------------------------------------
            # save the disparity image
            output_folder = os.path.join(args.savepath, "disparity", args.num_seq)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            fn = os.path.join(output_folder, fn.split('/')[-1].split('.')[0] + '.jpg')

            print("saving to", fn, disp_est.shape)
            if float(args.colored) == 1:
                disp_est = kitti_colormap(disp_est)
                cv2.imwrite(fn, disp_est)
            else:
                disp_est = np.round(disp_est * 256).astype(np.uint16)
                io.imsave(fn, disp_est)
            # -------------------------------------------------------------------------------------------------------------
    print("Done!")


@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


if __name__ == '__main__':
    test(args)
