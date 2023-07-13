

# Multimodal Interlaced Decoder

***ICCV 23***


## Environment

- Install MinkowskiEngine by using [Conda](https://github.com/NVIDIA/MinkowskiEngine#anaconda)

- Others

    Please refer to [env.yml](./env.yml) for details.

## Prepare data

- Download the dataset from official website.
    - [ScanNet](https://github.com/ScanNet/ScanNet)
    - [S3DIS](http://buildingparser.stanford.edu/dataset.html)

- 3D: dataset/preprocess_3d_scannet.py

## Config
- MIT_seg_2cm: config/scannet/3dunet_2cm_iccv_plabel.yaml


## Training

- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

NUMBER_OF_THREADS is the threads to use per process (gpu), so optimally, it should be **Total_threads / gpu_number_used**

## Testing

- Testing using your trained model or our [pre-trained model](https://drive.google.com/file/d/1bFiXViR0Pah7dK_Sa7Kw_2-mPB04oxRk/view?usp=sharing) (put the model weights in a model folder in `Exp/scannet/iccv_mink18_interlace/`):

```bash
sh tool/test.sh iccv_mink18_interlace Exp/scannet/iccv_mink18_interlace/3dunet_2cm_iccv_plabel.yaml 1)
```




## Acknowledgment

Our code is based on [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine). We also referred to [BPNet](https://github.com/wbhu/BPNet).



