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

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from src.dataset import HDF5Dataset, collate_fn_pad
from src.trainer import ModelWrapper

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
    output_dir = HydraConfig.get().runtime.output_dir
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
        
    torch_load_ckpt = torch.load(cfg.checkpoint)
    checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
    
    print(checkpoint_params.keys())
    print(checkpoint_params.cfg.exp_id)
    
    cfg.output = checkpoint_params.cfg.exp_id + f"-e{torch_load_ckpt['epoch']}-{cfg.av2_mode}-v{cfg.leaderboard_version}"
    cfg.model.update(checkpoint_params.cfg.model)
    
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
    print(f"\n---LOG[eval]: Loaded model from {cfg.checkpoint}. The backbone network is {checkpoint_params.cfg.model.name}.\n")

    wandb_logger = WandbLogger(save_dir=output_dir,
                               project=f"{cfg.wandb_project_name}", 
                               name=f"{cfg.output}",
                               offline=(cfg.wandb_mode == "offline"))
    
    trainer = pl.Trainer(logger=wandb_logger, devices=cfg.gpus)
    # NOTE(Qingwen): search & check: def eval_only_step_(self, batch, res_dict)
    

    print(cfg.dataset_path)
    
    val_loader = DataLoader(
        HDF5Dataset(cfg.dataset_path + "/val", 
                    n_frames=checkpoint_params.cfg.num_frames  if 'num_frames' in checkpoint_params.cfg else 2,
                    ),
        batch_size=1,
        shuffle=False,
    # collate_fn=collate_fn_pad,
    pin_memory=True)

    eval_loader = DataLoader(
        HDF5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}", 
                n_frames=checkpoint_params.cfg.num_frames  if 'num_frames' in checkpoint_params.cfg else 2,
                eval=True),
        batch_size=1,
        # collate_fn=collate_fn_pad,
        pin_memory=True,
        shuffle=False)
        
        
    print(f"---LOG[eval]: Start evaluation on {cfg.dataset_path}/{cfg.av2_mode}.")
    print(f"---LOG[eval]: Lenth of the eval data: {len(eval_loader)}, val data: {len(val_loader)}.")
    
    print(f"---LOG[eval]: Eval on {mymodel.av2_mode}.")
    trainer.validate(model = mymodel, 
                    dataloaders = eval_loader)
    
    mymodel.av2_mode = 'trainval'
    print("###"*20)
    print(f"---LOG[eval]: Eval on {mymodel.av2_mode}.")
    trainer.validate(model = mymodel, 
                    dataloaders = val_loader)
    wandb.finish()

if __name__ == "__main__":
    main()