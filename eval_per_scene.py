"""
# Created: 2023-08-09 10:28
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Output the evaluation results, go for local evaluation or online evaluation
"""
import pickle
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, ListConfig
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from src.dataset import HDF5Dataset, collate_fn_pad
from src.trainer import ModelWrapper
import os.path as osp
from tqdm import tqdm

def precheck_cfg_valid(cfg):
    if os.path.exists(cfg.dataset_path + f"/{cfg.av2_mode}") is False:
        raise ValueError(f"Dataset {cfg.dataset_path}/{cfg.av2_mode} does not exist. Please check the path.")
    if cfg.supervised_flag not in [True, False]:
        raise ValueError(f"Supervised flag {cfg.supervised_flag} is not valid. Please set it to True or False.")
    if cfg.leaderboard_version not in [1, 2]:
        raise ValueError(f"Leaderboard version {cfg.leaderboard_version} is not valid. Please set it to 1 or 2.")
    return cfg

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
        
    torch_load_ckpt = torch.load(cfg.checkpoint)
    checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
    
    exp_id = checkpoint_params.cfg.get('exp_id', cfg.model.name)
    print(exp_id)
    
    cfg.output = exp_id + f"-e{torch_load_ckpt['epoch']}-{cfg.av2_mode}-v{cfg.leaderboard_version}"
    cfg.model.update(checkpoint_params.cfg.model)
    
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
    print(f"\n---LOG[eval]: Loaded model from {cfg.checkpoint}. The backbone network is {checkpoint_params.cfg.model.name}.\n")

    
    output_dir = osp.join('output', exp_id, 'eval_per_scene')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if isinstance(cfg.gpus, ListConfig):
        assert len(cfg.gpus) == 1, "Only support single GPU for evaluation."
    else:
        cfg.gpus = 1 

    trainer = pl.Trainer(devices=cfg.gpus)
    # NOTE(Qingwen): search & check: def eval_only_step_(self, batch, res_dict)
    
    scene_list = pickle.load(open(osp.join(cfg.dataset_path, cfg.av2_mode, 'val_per_scene', 'scene_list.pkl'), 'rb'))

    scene_id = scene_list[4]
    print(cfg.dataset_path)
    print('scene_id:', scene_id)    

    for scene_id in scene_list:
        eval_loader = DataLoader(
            HDF5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}", 
                    n_frames=checkpoint_params.cfg.num_frames  if 'num_frames' in checkpoint_params.cfg else 2,
                    eval_per_scene=True,
                    scene_id=scene_id,
                    leaderboard_version=cfg.leaderboard_version),
            batch_size=1,
            # collate_fn=collate_fn_pad,
            # pin_memory=True,
            shuffle=False)
            
            
        print(f"---LOG[eval]: Start evaluation on {cfg.dataset_path}/{cfg.av2_mode}.")
        print(f"---LOG[eval]: Lenth of the eval data: {len(eval_loader)}.")
        
        print(f"---LOG[eval]: Eval on {mymodel.av2_mode}.")
        results = trainer.validate(model = mymodel, dataloaders = eval_loader)
        print(f"---LOG[eval]: Evaluation results: {results}.")

        pickle.dump(results, open(osp.join(output_dir, f'{scene_id}_results.pkl'), 'wb'))
        print('####'*20)
if __name__ == "__main__":
    main()