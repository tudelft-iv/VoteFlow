# SceneFlow by Translation Voting
## Goal
Incorporate the customized voting layer into ZeroFlow and use the cycle nn loss for unsupervised training.

## TODO
+ test the sf-voxel model

## Update Log
+ 2024.09.21 change to the codebase of [SeFlow](https://github.com/KTH-RPL/SeFlow)([README.md](./README_SeFlow.md))
+ 2024.09.19 update the sf-voxel model
+ 2024.07.04 build singularity environment on iv-mind ([image](https://surfdrive.surf.nl/files/index.php/s/BzXUog7XhThwRUf))
+ 2024.07.01 build a new branch based on [SceneFlowZoo](https://github.com/kylevedder/SceneFlowZoo)

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
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir data/Argoverse2 --output_dir data/Argoverse2/preprocess_v2
python dataprocess/extract_av2.py --av2_type sensor --data_mode val --mask_dir data/Argoverse2/3d_scene_flow
python dataprocess/extract_av2.py --av2_type sensor --data_mode test --mask_dir data/Argoverse2/3d_scene_flow
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
TODO

## Inference and Visualization
TODO

