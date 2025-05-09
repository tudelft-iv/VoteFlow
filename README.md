# VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow
Yancong Lin*, Shiming Wang*, Liangliang Nan, Julian Kooij, Holger Caesar

[![arXiv](https://img.shields.io/badge/arXiv-2503.22328-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.22328)

![](assets/docs/voteflow_arch.png)

## TODO
- [ ] Add more details to the README
- [ ] Update the trained weights
- [ ] Adapt the name of our method in the codebase

## Installation
```bash
conda env create -f environment.yaml
conda activate sf_tv
# CUDA already install in python environment. I also tested others version like 11.3, 11.4, 11.7, 11.8 all works
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

## Data Preprocess

```bash
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir /datasets/Argoverse2 --output_dir /datasets/Argoverse2/preprocess_v2 --nproc 24
python dataprocess/extract_av2.py --av2_type sensor --data_mode val --argo_dir /datasets/Argoverse2 --output_dir /datasets/Argoverse2/preprocess_v2 --mask_dir /datasets/Argoverse2/3d_scene_flow --nproc 24
python dataprocess/extract_av2.py --av2_type sensor --data_mode test --argo_dir /datasets/Argoverse2 --output_dir /datasets/Argoverse2/preprocess_v2 --mask_dir /datasets/Argoverse2/3d_scene_flow --nproc 24
```

## Related work (and codebases):
### Unsupervised
+ [ICP-FLOW](https://github.com/yanconglin/ICP-Flow)
+ [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/tree/main)
+ [ZeroFlow](https://github.com/kylevedder/zeroflow)
+ [SeFlow](https://github.com/KTH-RPL/SeFlow)
### Supervised

+ [DeFlow](https://arxiv.org/abs/2401.16122): ICRA 2024
+ [FastFlow3d](https://arxiv.org/abs/2103.01306): RA-L 2021

## Train 
``` python 
python train.py model=sf_voxel_model lr=2e-4 epochs=20 batch_size=2 loss_fn=warpedLoss
```
training on the complete dataset on 4 gpus

```python
python train.py model=sf_voxel_model lr=1e-3 epochs=20 batch_size=2 loss_fn=warpedLoss gpus=[0,1,2,3] wandb_mode=online exp_note="the_special_description_for_this_experiment"
```

```python
python train.py model=deflow lr=2e-4 epochs=20 batch_size=2 loss_fn=deflowLoss
```
## Evaluation 

```python
python eval.py checkpoint=logs/jobs/sf_voxel_model-0/09-27-09-33/checkpoints/17_sf_voxel_model.ckpt av2_mode=val
```
## Inference and Visualization
save the inference results into the demo data path
```python 
python save.py checkpoint=logs/jobs/sf_voxel_model-0/09-27-09-33/checkpoints/17_sf_voxel_model.ckpt dataset_path=data/Argoverse2_demo/preprocess_v2/sensor/val_vis res_name=sf_voxel_model
```

visualize with our tool

```python
python o3d_visualization.py index=17 res_name=sf_voxel_model  
```

## Acknowledgements
This code is mainly based on the [SeFlow](https://github.com/KTH-RPL/SeFlow) code by Qingwen Zhang. Thanks for her great work and codebase.
<!-- Additionally, we would like to express our sincere thanks to Kyle Vedder et al. for hosting and providing extensive support for Argoverse2 2024 Scene Flow Challenge -->