
"""
# Created: 2023-07-18 15:08
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""

import torch.nn as nn
import dztimer, torch
import pickle

from .basic.unet import FastFlow3DUNet
from .basic.encoder import DynamicEmbedder
from .basic.decoder import LinearDecoder, ConvGRUDecoder
from .basic import cal_pose0to1

class DeFlow(nn.Module):
    def __init__(self, voxel_size = [0.2, 0.2, 6],
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3],
                 grid_feature_size = [512, 512],
                 decoder_option = "gru",
                 num_iters = 4):
        super().__init__()
        self.embedder = DynamicEmbedder(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=32)
        
        self.backbone = FastFlow3DUNet()
        if decoder_option == "gru":
            self.head = ConvGRUDecoder(num_iters = num_iters)
        elif decoder_option == "linear":
            self.head = LinearDecoder()

        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, batch):
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        self.timer[0].start("Data Preprocess")
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        transform_pc0s = []
        for batch_id in range(batch_sizes):
            selected_pc0 = batch["pc0"][batch_id]
            # print(f"batch_id{batch_id}, pc0: {selected_pc0.shape}")
            # print(f'batch_id{batch_id},pc1: {batch["pc1"][batch_id].shape}')
            self.timer[0][0].start("pose")
            with torch.no_grad():
                if 'ego_motion' in batch:
                    pose_0to1 = batch['ego_motion'][batch_id]
                else:
                    pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            # transform selected_pc0 to pc1
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            self.timer[0][1].stop()
            pose_flows.append(transform_pc0 - selected_pc0)
            transform_pc0s.append(transform_pc0)

        pc0s = torch.stack(transform_pc0s, dim=0)
        pc1s = batch["pc1"]
        
        # print('pc0s:', pc0s.shape)
        # print('pc1s:', pc1s.shape)  
        self.timer[0].stop()

        self.timer[1].start("Voxelization")
        pc0_before_pseudoimages, pc0_voxel_infos_lst = self.embedder(pc0s)
        pc1_before_pseudoimages, pc1_voxel_infos_lst = self.embedder(pc1s)
        self.timer[1].stop()

        self.timer[2].start("Encoder")
        grid_flow_pseudoimage = self.backbone(pc0_before_pseudoimages,
                                            pc1_before_pseudoimages)
        self.timer[2].stop()

        self.timer[3].start("Decoder")
        print('Deflow backbone is called.')
        flows = self.head(
            torch.cat((pc0_before_pseudoimages, pc1_before_pseudoimages),
                    dim=1), grid_flow_pseudoimage, pc0_voxel_infos_lst)
        self.timer[3].stop()
        print('Deflow head is called.')
        # print("flows shape: ", len(flows), flows[0].shape) 
        pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
        pc1_points_lst = [e["points"] for e in pc1_voxel_infos_lst]

        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]
        pc1_valid_point_idxes = [e["point_idxes"] for e in pc1_voxel_infos_lst]

        # print('flow:', len(flows), flows[0].shape, flows[1].shape)
        # print('pose_flow:', len(pose_flows), pose_flows[0].shape, pose_flows[1].shape)
        # print('pc0_points_lst:', len(pc0_points_lst), pc0_points_lst[0].shape, pc0_points_lst[1].shape)
        # print('pc1_points_lst:', len(pc1_points_lst), pc1_points_lst[0].shape, pc1_points_lst[1].shape)
        model_res = {
            "flow": flows,
            'pose_flow': pose_flows,

            "pc0_valid_point_idxes": pc0_valid_point_idxes,
            "pc0_points_lst": pc0_points_lst,
            
            "pc1_valid_point_idxes": pc1_valid_point_idxes,
            "pc1_points_lst": pc1_points_lst,
        }
        
        ## save_results
        # vis_batch = dict(
        #     inputs = batch,
        #     outputs = model_res
        # )
        
        # file_path = 'outputs/vis_batch.pkl'
        # pickle.dump(vis_batch, open(file_path, 'wb'))
        
        return model_res