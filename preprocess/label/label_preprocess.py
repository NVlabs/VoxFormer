"""
Code partly taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/data/labels_downscale.py
"""
import numpy as np
from tqdm import tqdm
import numpy.matlib
import os
import glob
import io_data as SemanticKittiIO
import argparse
import yaml

def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


def majority_pooling(grid, k_size=2):
    result = np.zeros(
        (grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size)
    )
    for xx in range(0, int(np.floor(grid.shape[0] / k_size))):
        for yy in range(0, int(np.floor(grid.shape[1] / k_size))):
            for zz in range(0, int(np.floor(grid.shape[2] / k_size))):

                sub_m = grid[
                    (xx * k_size) : (xx * k_size) + k_size,
                    (yy * k_size) : (yy * k_size) + k_size,
                    (zz * k_size) : (zz * k_size) + k_size,
                ]
                unique, counts = np.unique(sub_m, return_counts=True)
                if True in ((unique != 0) & (unique != 255)):
                    # Remove counts with 0 and 255
                    counts = counts[((unique != 0) & (unique != 255))]
                    unique = unique[((unique != 0) & (unique != 255))]
                else:
                    if True in (unique == 0):
                        counts = counts[(unique != 255)]
                        unique = unique[(unique != 255)]
                value = unique[np.argmax(counts)]
                result[xx, yy, zz] = value
    return result



def main(config):
    scene_size = (256, 256, 32)
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    remap_lut = SemanticKittiIO._get_remap_lut(
        os.path.join(
            "./label/semantic-kitti.yaml",
        )
    )

    for sequence in sequences:
        sequence_path = os.path.join(
            config.kitti_root, "dataset", "sequences", sequence
        )
        label_paths = sorted(
            glob.glob(os.path.join(sequence_path, "voxels", "*.label"))
        )
        invalid_paths = sorted(
            glob.glob(os.path.join(sequence_path, "voxels", "*.invalid"))
        )
        out_dir = os.path.join(config.kitti_preprocess_root, "labels", sequence)
        os.makedirs(out_dir, exist_ok=True)

        downscaling = {"1_1": 1, "1_2": 2}

        for i in tqdm(range(len(label_paths))):

            frame_id, extension = os.path.splitext(os.path.basename(label_paths[i]))

            LABEL = SemanticKittiIO._read_label_SemKITTI(label_paths[i])
            INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_paths[i])
            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
                np.float32
            )  # Remap 20 classes semanticKITTI SSC
            LABEL[
                np.isclose(INVALID, 1)
            ] = 255  # Setting to unknown all voxels marked on invalid mask...
            LABEL = LABEL.reshape([256, 256, 32])

            for scale in downscaling:
                filename = frame_id + "_" + scale + ".npy"
                label_filename = os.path.join(out_dir, filename)
                # If files have not been created...
                if not os.path.exists(label_filename):
                    if scale != "1_1":
                        LABEL_ds = _downsample_label(
                            LABEL, (256, 256, 32), downscaling[scale]
                        )
                    else:
                        LABEL_ds = LABEL
                    np.save(label_filename, LABEL_ds)
                    print("wrote to", label_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./label_preprocess.py")
    parser.add_argument(
        '--kitti_root',
        '-r',
        type=str,
        required=True,
        help='kitti_root',
    )

    parser.add_argument(
        '--kitti_preprocess_root',
        '-p',
        type=str,
        required=True,
        help='kitti_preprocess_root',
    )
    config, unparsed = parser.parse_known_args()
    main(config)
