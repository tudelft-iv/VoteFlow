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
from omegaconf import DictConfig, ListConfig
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
    
    if cfg.use_demo_data:
        cfg.dataset_path = 'data/Argoverse2_demo/preprocess_v2/sensor'
        cfg.wandb_mode = 'offline'
        
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
        
    torch_load_ckpt = torch.load(cfg.checkpoint)
    checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
    
    # print(checkpoint_params.keys())

    exp_id = checkpoint_params.cfg.get('exp_id', cfg.model.name)
    # print(exp_id)
    
    cfg.output = exp_id + f"-e{torch_load_ckpt['epoch']}-{cfg.av2_mode}-v{cfg.leaderboard_version}"
    cfg.model.update(checkpoint_params.cfg.model)
    # print(type(cfg))
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
    print(f"\n---LOG[eval]: Loaded model from {cfg.checkpoint}. The backbone network is {checkpoint_params.cfg.model.name}.\n")

    wandb_logger = WandbLogger(save_dir=output_dir,
                               project=f"{cfg.wandb_project_name}", 
                               name=f"{cfg.output}",
                               offline=(cfg.wandb_mode == "offline"),
                               log_model=(True if cfg.wandb_mode == 'online' else False))
    # print(type(cfg.gpus))
    if isinstance(cfg.gpus, ListConfig):
        assert len(cfg.gpus) == 1, "Only support single GPU for evaluation."
    else:
        cfg.gpus = 1 
        
    # print(cfg.gpus)
    trainer = pl.Trainer(logger=wandb_logger, devices=cfg.gpus)
    # NOTE(Qingwen): search & check: def eval_only_step_(self, batch, res_dict)
        
    print(f"---LOG[eval]: Start evaluation on {cfg.dataset_path}/{cfg.av2_mode}.")
    if 'waymo' in cfg.dataset_path:
        cfg.with_trainval  = True
        print(f"---LOG[eval]: Eval on waymo dataset.")
    else:
        eval_loader = DataLoader(
            HDF5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}", 
                n_frames=checkpoint_params.cfg.num_frames  if 'num_frames' in checkpoint_params.cfg else 2,
                eval=True,
                using_pwpp_gm=cfg.using_pwpp_gm,
                leaderboard_version=cfg.leaderboard_version),
            batch_size=1,
            # collate_fn=collate_fn_pad,
            # pin_memory=True,
            shuffle=False)
        trainer.validate(model = mymodel, dataloaders = eval_loader)
        
        print(f"---LOG[eval]: Lenth of the eval data: {len(eval_loader)}.")
        print(f"---LOG[eval]: Eval on {mymodel.av2_mode}.")
    
    if cfg.with_trainval:
        val_loader = DataLoader(
        HDF5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}", #"/val", 
                    n_frames=checkpoint_params.cfg.num_frames  if 'num_frames' in checkpoint_params.cfg else 2,
                    ),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_pad,
        pin_memory=True)
            
        mymodel.av2_mode = 'trainval'
        print("###"*20)
        print(f"---LOG[eval]: Lenth of the val data: {len(val_loader)}.")
        print(f"---LOG[eval]: Eval on {mymodel.av2_mode}.")
        trainer.validate(model = mymodel, 
                        dataloaders = val_loader)
    wandb.finish()

if __name__ == "__main__":
    main()