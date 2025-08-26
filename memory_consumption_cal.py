import os
os.chdir('/home/shimingwang/workspace/sf_tv/sceneflow_tv_seflow_codebase')

import os.path as osp
import numpy as np
import h5py, pickle
from collections import defaultdict
from typing import Final
from pathlib import Path

from av2.structures.sweep import Sweep
from av2.datasets.sensor.av2_sensor_dataloader import convert_pose_dataframe_to_SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import CuboidList, Cuboid
from av2.utils.io import read_feather

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.dataset import HDF5Dataset, collate_fn_pad
from src.models.voteflow import VoteFlow

device = torch.device('cuda:0')
free, total = torch.cuda.mem_get_info(device)
mem_used_mb = (total - free) / 1024 ** 2
print('originak mem consumption:', mem_used_mb)

val_loader = DataLoader(HDF5Dataset('data/Argoverse2_demo/preprocess_v2/sensor/val', 
                                    n_frames=2),
                        batch_size=1,
                        shuffle=False,
                        num_workers=2,
                        collate_fn=collate_fn_pad,
                        pin_memory=True)


model = VoteFlow(m=16, n=128, use_bn_in_vol=True)
model = model.to(device)
model.eval()
for idx, batch in enumerate(val_loader):
    # print(batch['pc0'].shape)

    batch['pc0'] = batch['pc0'][:, :80000,:].to(device)
    
    print(batch['pc0'].shape)
    pc0 = torch.rand(1, 70000, 3).to(device)
    batch['pc0'] = pc0
    print(batch['pc0'].shape)
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)  
        elif isinstance(batch[key], list):
            new_list = []
            for ele in batch[key]:
                ele = ele.to(device)
                new_list.append(ele)
            batch[key] = new_list

    outs = model(batch)
    free, total = torch.cuda.mem_get_info(device)
    mem_used_mb = (total - free) / 1024 ** 2
    print(mem_used_mb)