# VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow (CVPR'25)
Yancong Lin*, Shiming Wang*, Liangliang Nan, Julian Kooij, Holger Caesar

[![arXiv](https://img.shields.io/badge/arXiv-2503.22328-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.22328)
[![YouTube Video Views](https://img.shields.io/youtube/views/38dQqMKGEEg)](https://www.youtube.com/watch?v=38dQqMKGEEg&t=163s&ab_channel=IntelligentVehiclesatTUDelft)
[![CVPR2025](https://img.shields.io/badge/CVPR-2025-blue)](https://surfdrive.surf.nl/files/index.php/s/uJWqDaSKdrMRh6U)

[Video Download](https://surfdrive.surf.nl/files/index.php/s/x5ssujp4J9VL63p) | 
[Poster Download](https://surfdrive.surf.nl/files/index.php/s/uJWqDaSKdrMRh6U) | 
[Pretrained Weights(m8n128) Download](https://surfdrive.surf.nl/files/index.php/s/58QM2FeoSANooi5)

## Notice
**Our method VoteFlow has been integrated in to the OpenSceneFlow.** Please check [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow) for the lastest updates and developments.

## Installation
```bash
conda env create -f environment.yaml
conda activate sf_tv
# CUDA already install in python environment. I also tested others version like 11.3, 11.4, 11.7, 11.8 all works
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

## Data Preprocess
Please follow the instructions of [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow) to process data.
```bash
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir /datasets/Argoverse2 --output_dir /datasets/Argoverse2/preprocess_v2 --nproc 24
python dataprocess/extract_av2.py --av2_type sensor --data_mode val --argo_dir /datasets/Argoverse2 --output_dir /datasets/Argoverse2/preprocess_v2 --mask_dir /datasets/Argoverse2/3d_scene_flow --nproc 24
python dataprocess/extract_av2.py --av2_type sensor --data_mode test --argo_dir /datasets/Argoverse2 --output_dir /datasets/Argoverse2/preprocess_v2 --mask_dir /datasets/Argoverse2/3d_scene_flow --nproc 24
```

## Train 
training on the complete dataset on 4 gpus

```python
python train.py model=voteflow lr=2e-4 epochs=12 batch_size=4 model.target.use_bn_in_vol=True model.target.m=8 model.target.n=128 model.target.decoder_layers=4 model.target.use_separate_feats_voting=True wandb_mode=online gpus=[0,1,2,3] loss_fn=seflowLoss exp_note=with_seflowLoss_decoder_using_separate_feats_voting add_seloss="{chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" model.val_monitor=val/Dynamic/Mean
```

## Evaluation 

```python
python eval.py checkpoint=checkpoints/voteflow_best_m8n128_ori.pth av2_mode=val
```

## Inference and Visualization
save the inference results into the demo data path
```python 
python save.py checkpoint=checkpoints/voteflow_best_m8n128_ori.pth  res_name=voteflow
```

visualize with our tool

```python
python o3d_visualization.py index=17 res_name=voetflow  
```

## Cite us
```bibtex
@inproceedings{lin2025voteflow,
  title={VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow},
  author={Lin, Yancong and Wang, Shiming and Nan, Liangliang and Kooij, Julian and Caesar, Holger},
  booktitle={CVPR},
  year={2025},
}
```

## Acknowledgements
This code is mainly based on the [SeFlow](https://github.com/KTH-RPL/SeFlow) code by Qingwen Zhang. For more instructions and functions, please refer to her original code. Thanks for her great work and codebase.