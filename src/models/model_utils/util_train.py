import argparse
import logging
import os
import copy
import os
import os.path as osp
import shutil
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def trainer(dataset, model, loss_fn, device):

    for param in model.parameters():
        param.requires_grad = True
    print("Using Adam optimizer.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    # optimizer.param_groups[0]['capturable'] = True

    losses = []
    loss_best = 1e8
    epoch_best = 0 
    mem_max = 0.0
    for epoch in range(1000):
        optimizer.zero_grad()

        loss_sum = 0.0
        num_sum = 0
        for k, data in enumerate(dataset):
            points_src = data['point_src']
            points_dst = data['point_dst']
            flows_gt = data['scene_flow']
            points_src = torch.from_numpy(points_src).float().to(device)
            points_dst = torch.from_numpy(points_dst).float().to(device)
            flows_gt = torch.from_numpy(flows_gt).float().to(device)
            # print('inputs: ', points_src.shape, points_dst.shape, flows.shape)

            flows_pred, masks_src, masks_dst = model(points_src, points_dst)
            # print('model output: ', flows_pred.shape, masks_src.shape, masks_dst.shape)

            loss = loss_fn(points_src, points_dst, flows_pred, flows_gt, masks_src, masks_dst)

            loss_sum += loss.item()
            num_sum += len(flows_pred)
            loss.backward()
            optimizer.step()

        if mem_max < torch.cuda.max_memory_reserved()/1024/1024/1024:
            mem_max = torch.cuda.max_memory_reserved()/1024/1024/1024
            print('cuda memory max (in GB): ', mem_max) 

        # ANCHOR: get best metrics
        loss_epoch = loss_sum / num_sum
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_epoch:.6f} ")

            if loss_epoch <= loss_best:
                loss_best = loss_epoch
                epoch_best = loss_epoch
                # save checkpoin best
                torch.save(
                {
                    "epoch": epoch_best,
                    "loss": loss_epoch,
                    "optim_state_dict": optimizer.state_dict(),
                    "model_state_dict": model.state_dict(),
                },
                osp.join("checkpoint_best.pth"),
                )
        losses.append(loss_epoch)

    # save checkpoin latest
    torch.save(
    {
        "epoch": epoch,
        "loss": loss_epoch,
        "optim_state_dict": optimizer.state_dict(),
        "model_state_dict": model.state_dict(),
    },
    osp.join("checkpoint_latest.pth"),
    )

    # # NOTE: visualization
    # fig = plt.figure(figsize=(13, 5))
    # ax = fig.gca()
    # ax.plot(total_losses, label="loss")
    # ax.legend(fontsize="14")
    # ax.set_xlabel("Iteration", fontsize="14")
    # ax.set_ylabel("Loss", fontsize="14")
    # ax.set_title("Loss vs iterations", fontsize="14")
    # plt.show()
