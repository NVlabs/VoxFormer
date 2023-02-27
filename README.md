<div align="center">   
  
# VoxFormer: a Cutting-edge Baseline for 3D Semantic Occupancy Prediction
</div>

![](https://img.shields.io/badge/Ranked%20%231-Camera--Only%203D%20SSC%20on%20SemanticKITTI-green "")

![](./teaser/scene08_13_19.gif "")

> **VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion**, CVPR 2023.

> [Yiming Li](https://scholar.google.com/citations?hl=en&user=i_aajNoAAAAJ&view_op=list_works&sortby=pubdate), [Zhiding Yu](https://scholar.google.com/citations?user=1VI_oYUAAAAJ&hl=en), [Chris Choy](https://scholar.google.com/citations?user=2u8G5ksAAAAJ&hl=en), [Chaowei Xiao](https://scholar.google.com/citations?user=Juoqtj8AAAAJ&hl=en), [Jose M. Alvarez](https://scholar.google.com/citations?user=Oyx-_UIAAAAJ&hl=en), [Sanja Fidler](https://scholar.google.com/citations?user=CUlqK5EAAAAJ&hl=en), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ&hl=en), [Anima Anandkumar](https://scholar.google.com/citations?user=bEcLezcAAAAJ&hl=en)

>  [[PDF]](https://arxiv.org/pdf/2302.12251.pdf) [[Project]](https://github.com/NVlabs/VoxFormer) 


# News
- [2023/02]: VoxFormer is accepted by CVPR 2023.
- [2023/02]: Our paper is on [arxiv](https://arxiv.org/abs/2302.12251).
- [2022/11]: VoxFormer achieve the SOTA on [SemanticKITTI 3D SSC (Semantic Scene Completion) Task](http://www.semantic-kitti.org/tasks.html#ssc) with **13.35% mIoU** and **44.15% IoU** (camera-only)!
</br>


# Abstract
Humans can easily imagine the complete 3D geometry of occluded objects and scenes. This appealing ability is vital for recognition and understanding. To enable such capability in AI systems, we propose VoxFormer, a Transformer-based semantic scene completion framework that can output complete 3D volumetric semantics from only 2D images. Our framework adopts a two-stage design where we start from a sparse set of visible and occupied voxel queries from depth estimation, followed by a densification stage that generates dense 3D voxels from the sparse ones. A key idea of this design is that the visual features on 2D images correspond only to the visible scene structures rather than the occluded or empty spaces. Therefore, starting with the featurization and prediction of the visible structures is more reliable. Once we obtain the set of sparse queries, we apply a masked autoencoder design to propagate the information to all the voxels by self-attention. Experiments on SemanticKITTI show that VoxFormer outperforms the state of the art with a relative improvement of 20.0% in geometry and 18.1% in semantics and reduces GPU memory during training by ~45% to less than 16GB.


# Method

| ![space-1.jpg](teaser/arch.png) | 
|:--:| 
| ***Figure 1. Overall framework of VoxFormer**. Given RGB images, 2D features are extracted by ResNet50 and the depth is estimated by an off-the-shelf depth predictor. The estimated depth after correction enables the class-agnostic query proposal stage: the query located at an occupied position will be selected to carry out deformable cross-attention with image features. Afterwards, mask tokens will be added for completing voxel features by deformable self-attention. The refined voxel features will be upsampled and projected to the output space for per-voxel semantic segmentation. Note that our framework supports the input of single or multiple images.* |


# Dataset

- [ ] nuScenes
- [ ] KITTI-360
- [x] SemanticKITTI


# Bibtex
If this work is helpful for your research, please cite the following BibTeX entry.

```
@misc{li2023voxformer,
      title={VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion}, 
      author={Yiming Li and Zhiding Yu and Christopher Choy and Chaowei Xiao and Jose M. Alvarez and Sanja Fidler and Chen Feng and Anima Anandkumar},
      year={2023},
      eprint={2302.12251},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement

Many thanks to these excellent open source projects:
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
