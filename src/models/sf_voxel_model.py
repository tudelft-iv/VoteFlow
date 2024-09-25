import math
import numpy as np

import dztimer
import torch
import torch.nn as nn
from torch_scatter import scatter_max
import pytorch3d.ops as pytorch3d_ops

from .basic.encoder import DynamicEmbedder
from .basic import cal_pose0to1

from .ht.ht_cuda import HT_CUDA
from .model_utils.util_model import Backbone, VolConv, Decoder, FastFlowUNet
from .model_utils.util_func import float_division, tensor_mem_size, calculate_unq_voxels, batched_masked_gather, pad_to_batch

import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

class SFVoxelModel(nn.Module):
    def __init__(self, 
                 nframes=1, 
                 m=8, 
                 n=64, 
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3], 
                 voxel_size=(0.2, 0.2, 0.2),
                 grid_feature_size = [512, 512],
                 **kwargs):
        super(SFVoxelModel, self).__init__()
        
        assert len(point_cloud_range)==6
        assert len(voxel_size)==3

        # a hack there, may and may not have any impact depends on the setting.
        assert voxel_size[0]==voxel_size[1]

        # how many bins inside a voxel after quantization
        # assume 120km/h, along x/y; 0.1m along z
        nx = math.ceil(3.3*nframes / voxel_size[0])  
        ny = math.ceil(3.3*nframes / voxel_size[1]) 
        nz = math.ceil(0.1 / voxel_size[2])  # +/-0.1

        self.nx = nx*2 # +/-x
        self.ny = ny*2 # +/-y
        self.nz = nz*2 # +/-z
        print('n x/y/z: ', self.nx, self.ny, self.nz)

        pseudo_image_dims = grid_feature_size[:2] #int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0]), int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1])) # ignore z dimension
    
        self.nframes = nframes
        self.m = m # m knn within src, for each src point
        self.radius_src = math.ceil(max(2.0/voxel_size[0], 2.0/voxel_size[1])) # define a search window (in meters) within src voxels, aka the rigid motion window
        self.n = n # n knn between src and dst, for each src voxel
        self.radius_dst = max(self.nx, self.ny) # define a search window for a src voxel in dst voxels for calculating translations
        print('radius: ', self.radius_src, self.radius_dst)

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.pseudo_image_dims = pseudo_image_dims
        
        self.backbone = Backbone(32, 16)
        # self.backbone = FastFlowUNet(32, 16)
        self.vote = HT_CUDA(self.ny, self.nx, self.nz)
        self.volconv = VolConv(self.ny, self.nx, self.nz, c=16*2, dim_output=16)
        self.decoder = Decoder(dim_input=32*2+16+16, dim_output=3)

        self.embedder = DynamicEmbedder(voxel_size=voxel_size,
                                pseudo_image_dims=pseudo_image_dims,
                                point_cloud_range=point_cloud_range,
                                feat_channels=32)
        
        self.timer = dztimer.Timing()
        

    def process_points_per_pair(self, voxel_info_src, voxel_info_dst):
        valid_point_idxs_src = voxel_info_src['point_idxes']
        valid_point_idxs_dst = voxel_info_dst['point_idxes']
        valid_voxel_coords_src = voxel_info_src['voxel_coords']
        valid_voxel_coords_dst = voxel_info_dst['voxel_coords']

        unq_voxel_coords_src, point_voxel_idxs_src = calculate_unq_voxels(valid_voxel_coords_src, self.pseudo_image_dims)
        unq_voxel_coords_dst, point_voxel_idxs_dst = calculate_unq_voxels(valid_voxel_coords_dst, self.pseudo_image_dims)

        dists_dst, knn_idxs_dst, _ = pytorch3d_ops.ball_query(unq_voxel_coords_src[None].float(), unq_voxel_coords_dst[None].float(), lengths1=None, lengths2=None, K=self.n, return_nn=False, radius=self.radius_dst)
        dists_src, knn_idxs_src, _ = pytorch3d_ops.ball_query(unq_voxel_coords_src[None].float(), unq_voxel_coords_src[None].float(), lengths1=None, lengths2=None, K=self.m, return_nn=False, radius=self.radius_src)
        
        return valid_point_idxs_src, valid_point_idxs_dst, point_voxel_idxs_src, point_voxel_idxs_dst, unq_voxel_coords_src, unq_voxel_coords_dst, knn_idxs_src[0], knn_idxs_dst[0]
                
    def preprocessing(self, points_src, points_dst, voxel_info_list_src, voxel_info_list_dst):
        point_masks_src = []
        point_masks_dst = []
        point_voxel_idxs_src = []
        point_voxel_idxs_dst = []
        unq_voxels_src = []
        unq_voxels_dst = []
        knn_idxs_src = []
        knn_idxs_dst = []

        l_points = 0 # for padding
        l_voxels = 0 # for padding
        for points_src_, points_dst_, voxel_info_src, voxel_info_dst in zip(points_src, points_dst, voxel_info_list_src, voxel_info_list_dst):
            point_idxs_src_, point_idxs_dst_, \
                point_voxel_idxs_src_, point_voxel_idxs_dst_, \
                    unq_voxels_src_, unq_voxels_dst_,  \
                        knn_idxs_src_, knn_idxs_dst_ = self.process_points_per_pair(voxel_info_src, voxel_info_dst)
            # print('process_points_per_pair: ', point_idxs_src_.shape, point_voxel_idxs_src_.shape, unq_voxels_src_.shape, knn_idxs_dst_.shape)

            # generate a mask for easy batching and sequential processing
            point_masks_src_ = torch.zeros((len(points_src_)), device=points_src_.device, dtype=points_src_.dtype)
            point_masks_src_[point_idxs_src_] = 1
            point_masks_dst_ = torch.zeros((len(points_dst_)), device=points_dst_.device, dtype=points_dst_.dtype)
            point_masks_dst_[point_idxs_dst_] = 1
            point_masks_src.append(point_masks_src_)
            point_masks_dst.append(point_masks_dst_)

            point_voxel_idxs_src.append(point_voxel_idxs_src_)
            point_voxel_idxs_dst.append(point_voxel_idxs_dst_)
            unq_voxels_src.append(unq_voxels_src_)
            unq_voxels_dst.append(unq_voxels_dst_)
            knn_idxs_src.append(knn_idxs_src_)
            knn_idxs_dst.append(knn_idxs_dst_)

            l_voxels = max(l_voxels, max(unq_voxels_src_.shape[0], unq_voxels_dst_.shape[0]))
            l_points = max(l_points, max(point_voxel_idxs_src_.shape[0], point_voxel_idxs_dst_.shape[0]))
        # print('l', l_points, l_voxels)

        # padding
        for i, (point_voxel_idxs_src_, point_voxel_idxs_dst_, unq_voxels_src_, unq_voxels_dst_, knn_idxs_src_, knn_idxs_dst_) \
            in enumerate( zip(point_voxel_idxs_src, point_voxel_idxs_dst, unq_voxels_src, unq_voxels_dst, knn_idxs_src, knn_idxs_dst) ):
            unq_voxels_src_= pad_to_batch(unq_voxels_src_, l_voxels)
            unq_voxels_dst_ = pad_to_batch(unq_voxels_dst_, l_voxels)
            knn_idxs_src_ = pad_to_batch(knn_idxs_src_, l_voxels)
            knn_idxs_dst_= pad_to_batch(knn_idxs_dst_, l_voxels)
            point_voxel_idxs_src_= pad_to_batch(point_voxel_idxs_src_, l_points)
            point_voxel_idxs_dst_= pad_to_batch(point_voxel_idxs_dst_, l_points)

            unq_voxels_src[i] = unq_voxels_src_
            unq_voxels_dst[i] = unq_voxels_dst_
            knn_idxs_src[i] = knn_idxs_src_
            knn_idxs_dst[i] = knn_idxs_dst_
            point_voxel_idxs_src[i] = point_voxel_idxs_src_
            point_voxel_idxs_dst[i] = point_voxel_idxs_dst_

        unq_voxels_src = torch.stack(unq_voxels_src, dim=0)
        unq_voxels_dst = torch.stack(unq_voxels_dst, dim=0)
        knn_idxs_src = torch.stack(knn_idxs_src, dim=0)
        knn_idxs_dst = torch.stack(knn_idxs_dst, dim=0)
        point_voxel_idxs_src = torch.stack(point_voxel_idxs_src, dim=0)
        point_voxel_idxs_dst = torch.stack(point_voxel_idxs_dst, dim=0)
        point_masks_src = torch.stack(point_masks_src, dim=0)
        point_masks_dst = torch.stack(point_masks_dst, dim=0)

        return point_masks_src, point_masks_dst, point_voxel_idxs_src, point_voxel_idxs_dst, unq_voxels_src, unq_voxels_dst, knn_idxs_src, knn_idxs_dst
    
    def extract_voxel_from_image(self, image, voxels):
        # image: [b, c, h, w]; voxels: [b, num, 2]
        idxs = voxels[:, :, 0] * self.pseudo_image_dims[0] + voxels[:, :, 1]
        b, c, h, w = image.shape
        feats_per_voxel = batched_masked_gather(image.view(b, c, h*w).permute(0, 2, 1), idxs.long(), idxs>=0, fill_value=-1)
        # print('point per voxel: ', feats_per_voxel.shape)
        return feats_per_voxel # [b, num , c]

    def extract_point_from_voxel(self, voxels, idxs):
        # voxels: [b, l, c]; idxs: [b, num]
        feats_per_point = batched_masked_gather(voxels, idxs.long(), idxs>=0, fill_value=-1)
        # print('point per point: ', feats_per_point.shape)
        return feats_per_point # [b, num, c]

    def extract_point_from_image(self, image, voxels, point_idxs):
        feats_per_voxel = self.extract_voxel_from_image(image, voxels)
        feats_per_point = self.extract_point_from_voxel(feats_per_voxel, point_idxs)
        # print('point per point merged: ', feats_per_point.shape)
        return feats_per_point

    # adapted from zeroflow
    def concat_feats(self, feats_vote, feats_before, feats_after, voxelizer_infos):
        feats = torch.cat([feats_vote, feats_before, feats_after], dim=-1)
        return feats

    def _model_forward(self, points_src, points_dst):
        # assert points_src.shape==points_dst.shape

        self.timer[1].start("Voxelization")
        pseudoimages_src, voxel_infos_lst_src = self.embedder(points_src)
        pseudoimages_dst, voxel_infos_lst_dst = self.embedder(points_dst)
        self.timer[1].stop()
        
        self.timer[2].start("Preprocessing")
        with torch.no_grad():
            point_masks_src, point_masks_dst, \
                point_voxel_idxs_src, point_voxel_idxs_dst, \
                    voxels_src, voxels_dst, \
                        knn_idxs_src, knn_idxs_dst = self.preprocessing(points_src, points_dst, voxel_infos_lst_src, voxel_infos_lst_dst)
        self.timer[2].stop()
        
        self.timer[3].start("Feature extraction")
        pseudoimages_grid = self.backbone(pseudoimages_src, pseudoimages_dst)
        feats_voxel_src = self.extract_voxel_from_image(pseudoimages_grid, voxels_src)
        feats_voxel_dst = self.extract_voxel_from_image(pseudoimages_grid, voxels_dst)
        self.timer[3].stop()
        
        self.timer[4].start("Voting")
        vols= self.vote(feats_voxel_src, feats_voxel_dst, voxels_src, voxels_dst, knn_idxs_src, knn_idxs_dst) 
        vols = self.volconv(vols)
        self.timer[4].stop()
        
        self.timer[5].start("Decoding")
        feats_point_src_vol = self.extract_point_from_voxel(vols, point_voxel_idxs_src)
        feats_point_src_init = self.extract_point_from_image(torch.cat([pseudoimages_src, pseudoimages_dst], dim=1), voxels_src, point_voxel_idxs_src)
        feats_point_src_grid = self.extract_point_from_image(pseudoimages_grid,  voxels_src, point_voxel_idxs_src)
        feats_cat = self.concat_feats(feats_point_src_vol, feats_point_src_init, feats_point_src_grid, voxel_infos_lst_src)
        flows = self.decoder(feats_cat)
        self.timer[5].stop()
        
        pc0_points_lst = [e["points"] for e in voxel_infos_lst_src]
        pc1_points_lst = [e["points"] for e in voxel_infos_lst_dst]
        
        pc0_valid_point_idxes = [e["point_idxes"] for e in voxel_infos_lst_src]
        pc1_valid_point_idxes = [e["point_idxes"] for e in voxel_infos_lst_dst]
        
        flows_reshape = []
        for (flow, valid_pts) in zip(flows, pc0_valid_point_idxes):
            flow = flow[:valid_pts.shape[0], :]
            flows_reshape.append(flow)
        # print('points_src', points_src.shape)
        # print('pc0_points_lst:', len(pc0_points_lst), pc0_points_lst[0].shape, pc0_points_lst[1].shape)
        # print('flow:', len(flows), flows[0].shape, flows[1].shape)
        # print('resahpe_flow:', len(flows_reshape), flows_reshape[0].shape, flows_reshape[1].shape)
        # print('point_masks_src:', point_masks_src.shape, sum(point_masks_src[0]), sum(point_masks_src[1]))

        model_res = {
            "flow": flows_reshape,
            "pc0_points_lst": pc0_points_lst,
            "pc1_points_lst": pc1_points_lst,
            "pc0_valid_point_idxes": pc0_valid_point_idxes,
            "pc1_valid_point_idxes": pc1_valid_point_idxes,
        }
        
        return model_res
    
    
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
        self.timer[0].stop()
        
        
        model_res = self._model_forward(pc0s, pc1s)
        
        ret_dict = model_res
        ret_dict["pose_flow"] = pose_flows
        
        return ret_dict
        