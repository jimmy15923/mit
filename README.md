# [ICCV'23] 2D-3D Interlaced Transformer for Point Cloud Segmentation with Scene-Level Supervision
[[ArXiv](https://arxiv.org/abs/2310.12817)][[Project page](https://jimmy15923.github.io/mit_web/)][[Video](https://www.youtube.com/watch?v=32nvE73SmuY)][[Poster]()]
[[Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_2D-3D_Interlaced_Transformer_for_Point_Cloud_Segmentation_with_Scene-Level_Supervision_ICCV_2023_paper.html)]

This repository is the implementation of our ICCV 2023 paper: **2D-3D Interlaced Transformer for Point Cloud Segmentation with Scene-Level Supervision**.

<p align="center">
<img src="figure/MIT_teaser.png?raw=true" width="70%">
</p>

## Requirements
Build the conda environment by
```bash
conda env create -f mit_env.yaml
```

We implement our MIT by using MinkowskiEngine. Please follow the installation instruction from [their GitHub](https://github.com/NVIDIA/MinkowskiEngine#anaconda). We also utilize the third-party [point cloud process](https://github.com/072jiajia/point-cloud-lib) library from Ji-Jia Wu.

## Data Preparation
Download the ScanNet [here](https://github.com/ScanNet/ScanNet).

- We follow [BPNet](https://github.com/wbhu/BPNet/tree/main/prepare_2d_data) to prepare the 2D and 3D data. 

- Donwload the unsupervised pre-computed supervoxel by [WYPR](https://github.com/facebookresearch/WyPR/blob/main/docs/RUNNING.md#shape-detection)

The data sctructure should be like:

```text
├── data_root
│   ├── train
│   │   ├── scene0000_00.pth
│   │   ├── scene0000_01.pth
│   │── val
│   │   ├── scene0011_00.pth
│   │   ├── scene0011_01.pth
│   ├── 2D
│   │   ├── scene0000_00
│   │   |   ├── color
│   │   |   ├── label
```


## Training
Start training: `sh tool/train.sh $EXP_NAME$ $/PATH/TO/CONFIG$ $NUMBER_OF_THREADS$`
```bash
sh tool/train.sh configs/ICCV23/config.yaml mit 8
```

## Acknowledgment

Our code is based on [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine). We also referred to [BPNet](https://github.com/wbhu/BPNet).


## Citation
If you find our work useful in your research, please consider citing our paper:
```
@inproceedings{yang20232d,
  title={2D-3D Interlaced Transformer for Point Cloud Segmentation with Scene-Level Supervision},
  author={Yang, Cheng-Kun and Chen, Min-Hung and Chuang, Yung-Yu and Lin, Yen-Yu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={977--987},
  year={2023}
}
```

