import math
import numpy as np

import dztimer
import torch
import torch.nn as nn
from torch_scatter import scatter_max
import pytorch3d.ops as pytorch3d_ops

from .basic.encoder import DynamicEmbedder
from .basic.decoder import LinearDecoder
from .basic import cal_pose0to1

from .ht.ht_cuda import HT_CUDA
from .model_utils.util_model import Backbone, VolConv, VolConvBN, Decoder, FastFlow3DUNet, SimpleDecoder, GRUDecoder, FastFlow3DDecoder, FastFlow3DDecoderWithProjHead
from .model_utils.util_func import float_division, tensor_mem_size, calculate_unq_voxels, batched_masked_gather, pad_to_batch

import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

class SFVoxelModel(nn.Module):
    def __init__(self, 
                 nframes=1, 
                 m=8, 
                 n=64, 
                 using_voting=True,
                 input_channels=32,
                 output_channels=64, 
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3], 
                 voxel_size=(0.2, 0.2, 6),
                 grid_feature_size = [512, 512],
                 decoder='decoder',
                 decoder_layers=1,
                 use_bn_in_vol=False,
                 use_ball_query=False,
                 use_separate_feats_voting=False,
                 vol_conv_hidden_dim=16,
                 **kwargs):
        super().__init__()
        
        assert len(point_cloud_range)==6
        assert len(voxel_size)==3

        # a hack there, may and may not have any impact depends on the setting.
        assert voxel_size[0]==voxel_size[1]
        self.using_voting = using_voting
        print('Using voting:', self.using_voting)
        
        # General architecture:
        pseudo_image_dims = grid_feature_size[:2] #int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0]), int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1])) # ignore z dimension
        self.decoder = decoder
                    
        print(f'using decoder: {self.decoder}')
        
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.pseudo_image_dims = pseudo_image_dims
        #self.backbone = Backbone(input_channels, output_channels)
        self.backbone = FastFlow3DUNet(input_channels, output_channels) ## output_channel 64
        
        if self.decoder == 'simple_decoder':
            assert self.using_voting
            self.decoder = SimpleDecoder(dim_input=output_channels)
            print('simple decoder:', self.decoder)
        elif self.decoder == 'gru_decoder':
            self.decoder = GRUDecoder(pseudoimage_channels=output_channels, using_voting=self.using_voting)
            print('gru decoder:', self.decoder)
        elif self.decoder == 'ff3d_decoder_proj':
            self.decoder = FastFlow3DDecoderWithProjHead(pseudoimage_channels=output_channels, using_voting=self.using_voting)
            print('ff3d decoder with projection head:', self.decoder)
        elif self.decoder == 'decoder':
            if self.using_voting:
                self.decoder = Decoder(dim_input= output_channels * 2 + input_channels * 2, layer_size=decoder_layers)
                print('decoder:', self.decoder)
            else:
                self.decoder = Decoder(dim_input= output_channels + input_channels * 2, layer_size=decoder_layers)
                print('decoder:', self.decoder)
        elif self.decoder == 'ff3d_decoder':
            if self.using_voting:
                self.decoder = FastFlow3DDecoder(dim_input= output_channels * 2 + input_channels*2)
                print('FastFlow3D decoder:', self.decoder)
            else: 
                self.decoder = FastFlow3DDecoder(dim_input= output_channels + input_channels*2)
                print('FastFlow3D decoder:', self.decoder)
        else:
            raise NotImplementedError(f"decoder {self.decoder} not implemented")
        
        self.embedder = DynamicEmbedder(voxel_size=voxel_size,
                                pseudo_image_dims=pseudo_image_dims,
                                point_cloud_range=point_cloud_range,
                                feat_channels=input_channels)
        
        # Voting Module settings
        if self.using_voting:
            ## how many bins inside a voxel after quantization
            ## assume 72km/h (20m/s), along x/y; 0.1m along z
            e = 1e-8
            radius = 2
            nx = math.ceil((radius+e)*nframes / voxel_size[0])  
            ny = math.ceil((radius+e)*nframes / voxel_size[1]) 
            nz = math.ceil((0.1+e) / voxel_size[2])  # +/-0.1
            
            self.nx = nx*2 # +/-x
            self.ny = ny*2 # +/-y
            self.nz = nz*2 # +/-z
            print('n x/y/z: ', self.nx, self.ny, self.nz)
                    
            self.nframes = nframes
            self.m = m # m knn within src, for each src point
            self.using_ball_query = use_ball_query
            self.n = n # n knn between src and dst, for each src voxel
            self.radius_dst = max(nx, ny) # define a search window for a src voxel in dst voxels for calculating translations

            if self.using_ball_query:
                self.radius_src = math.ceil(max((radius+e)/voxel_size[0], (radius+e)/voxel_size[1])) # define a search window (in meters) within src voxels, aka the rigid motion window
                print(f'using ball query to search window radius in source pc: {self.radius_src}, m={self.m};')
            else:
                print(f'using knn to search window radius in source pc, m={self.m};')
            print(f'using ball query to search in target pc, n={self.n}, search window radius: {self.radius_dst}.')
            self.use_separate_feats = use_separate_feats_voting
            print(f'using separate features for voting: {self.use_separate_feats}')

            self.vote = HT_CUDA(self.ny, self.nx, self.nz)
            
            if use_bn_in_vol:
                self.volconv = VolConvBN(self.ny, self.nx, hidden_dim=vol_conv_hidden_dim, dim_output=output_channels)
            else:
                self.volconv = VolConv(self.ny, self.nx, hidden_dim=vol_conv_hidden_dim, dim_output=output_channels)
        

        self.timer = dztimer.Timing()
        self.timer.start("Total")
        
    def process_points_per_pair(self, voxel_info_src, voxel_info_dst):
        valid_point_idxs_src = voxel_info_src['point_idxes'] # [N_valid_pts]
        valid_point_idxs_dst = voxel_info_dst['point_idxes']
        valid_voxel_coords_src = voxel_info_src['voxel_coords'] #[N_valid_pts, 3]
        valid_voxel_coords_dst = voxel_info_dst['voxel_coords']
        
        # print('keys in the voxel info:', voxel_info_src.keys())
        # print('valid_points_src:', voxel_info_src['points'].shape)
        # print('valid_points_dst:', voxel_info_dst['points'].shape)
        # print('valid_point_idxs_src:', valid_point_idxs_src.shape)
        # # print('valid_point_idxs_src_top_20:', valid_point_idxs_src[:20])
        # print('valid_point_idxs_dst:', valid_point_idxs_dst.shape)
        # print('valid_voxel_coords_src:', valid_voxel_coords_src.shape)
        # # print('valid_voxel_coords_src_top_20:', valid_voxel_coords_src[:20,:])
        # print('valid_voxel_coords_dst:', valid_voxel_coords_dst.shape)
        # print('pseudo_image_dims:', self.pseudo_image_dims)
        
        unq_voxel_coords_src, point_voxel_idxs_src = calculate_unq_voxels(valid_voxel_coords_src, self.pseudo_image_dims)
        unq_voxel_coords_dst, point_voxel_idxs_dst = calculate_unq_voxels(valid_voxel_coords_dst, self.pseudo_image_dims)

        # unq_voxel_coords_src # [N_valid_voxels, 2]
        # point_voxel_idxs_src # [N_valid_pts], the index of the unique voxel for each point    
        if self.using_voting:
            dists_dst, knn_idxs_dst, _ = pytorch3d_ops.ball_query(unq_voxel_coords_src[None].float(), unq_voxel_coords_dst[None].float(), lengths1=None, lengths2=None, K=self.n, return_nn=False, radius=self.radius_dst)
            if self.using_ball_query:
                dists_src, knn_idxs_src, _ = pytorch3d_ops.ball_query(unq_voxel_coords_src[None].float(), unq_voxel_coords_src[None].float(), lengths1=None, lengths2=None, K=self.m, return_nn=False, radius=self.radius_src)
            else:
                dists_src, knn_idxs_src, _ = pytorch3d_ops.knn_points(unq_voxel_coords_src[None].float(), unq_voxel_coords_src[None].float(), lengths1=None, lengths2=None, K=self.m, return_nn=False, return_sorted=False)
        else:
            knn_idxs_dst=[None]
            knn_idxs_src=[None]
        # print('unq_voxel_coords_src:', unq_voxel_coords_src.shape) # [N_valid_voxels, 2]
        # print('knn_idxs_dst:', knn_idxs_dst.shape) # [1, N_valid_voxels, n], n=64,  the index of n nerest voxels in dst for each voxel in src
        # print('knn_idxs_src:', knn_idxs_src.shape) # [1, N_valid_voxels, m], m=8, the index of n nerest voxels in src for each voxel in src

        return valid_point_idxs_src, valid_point_idxs_dst, point_voxel_idxs_src, point_voxel_idxs_dst, unq_voxel_coords_src, unq_voxel_coords_dst, knn_idxs_src[0], knn_idxs_dst[0]
                
    def preprocessing(self, points_src, points_dst, voxel_info_list_src, voxel_info_list_dst):
        point_masks_src = []
        point_masks_dst = []
        point_voxel_idxs_src = []
        point_voxel_idxs_dst = []
        point_offsets_src = []
        
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
            point_offsets_src.append(voxel_info_src['point_offsets'])
            unq_voxels_src.append(unq_voxels_src_)
            unq_voxels_dst.append(unq_voxels_dst_)
            knn_idxs_src.append(knn_idxs_src_)
            knn_idxs_dst.append(knn_idxs_dst_)

            l_voxels = max(l_voxels, max(unq_voxels_src_.shape[0], unq_voxels_dst_.shape[0]))
            l_points = max(l_points, max(point_voxel_idxs_src_.shape[0], point_voxel_idxs_dst_.shape[0]))
        # print('l', l_points, l_voxels)

        # padding
        for i, (point_voxel_idxs_src_, point_voxel_idxs_dst_, point_offsets_src_, unq_voxels_src_, unq_voxels_dst_, knn_idxs_src_, knn_idxs_dst_) \
            in enumerate( zip(point_voxel_idxs_src, point_voxel_idxs_dst, point_offsets_src, unq_voxels_src, unq_voxels_dst, knn_idxs_src, knn_idxs_dst) ):
            unq_voxels_src_= pad_to_batch(unq_voxels_src_, l_voxels)
            unq_voxels_dst_ = pad_to_batch(unq_voxels_dst_, l_voxels)
            point_voxel_idxs_src_= pad_to_batch(point_voxel_idxs_src_, l_points)
            point_voxel_idxs_dst_= pad_to_batch(point_voxel_idxs_dst_, l_points)
            point_offsets_src_ = pad_to_batch(point_offsets_src_, l_points)
            
            knn_idxs_src_ = pad_to_batch(knn_idxs_src_, l_voxels)
            knn_idxs_dst_= pad_to_batch(knn_idxs_dst_, l_voxels)

            unq_voxels_src[i] = unq_voxels_src_
            unq_voxels_dst[i] = unq_voxels_dst_
            point_voxel_idxs_src[i] = point_voxel_idxs_src_
            point_voxel_idxs_dst[i] = point_voxel_idxs_dst_
            point_offsets_src[i] = point_offsets_src_
            
            knn_idxs_src[i] = knn_idxs_src_
            knn_idxs_dst[i] = knn_idxs_dst_
            
        unq_voxels_src = torch.stack(unq_voxels_src, dim=0)
        unq_voxels_dst = torch.stack(unq_voxels_dst, dim=0)
        point_voxel_idxs_src = torch.stack(point_voxel_idxs_src, dim=0)
        point_voxel_idxs_dst = torch.stack(point_voxel_idxs_dst, dim=0)
        point_offsets_src = torch.stack(point_offsets_src, dim=0)
        point_masks_src = torch.stack(point_masks_src, dim=0)
        point_masks_dst = torch.stack(point_masks_dst, dim=0)
        
        if self.using_voting:
            knn_idxs_src = torch.stack(knn_idxs_src, dim=0)
            knn_idxs_dst = torch.stack(knn_idxs_dst, dim=0)
        else:
            knn_idxs_src = None
            knn_idxs_dst = None

        return point_masks_src, point_masks_dst, point_voxel_idxs_src, point_voxel_idxs_dst, point_offsets_src, unq_voxels_src, unq_voxels_dst, knn_idxs_src, knn_idxs_dst
    
    def extract_voxel_from_image(self, image, voxels):
        # image: [b, c, h, w]; voxels: [b, num, 2]
        idxs = voxels[:, :, 0] * self.pseudo_image_dims[0] + voxels[:, :, 1]
        
        mask = voxels.min(-1)[0]>=0
        
        b, c, h, w = image.shape
        
        feats_per_voxel = batched_masked_gather(image.view(b, c, h*w).permute(0, 2, 1), idxs[:, :, None].long(), mask[:, :, None], fill_value=0)
        
        # print('point per voxel: ', feats_per_voxel.shape)
        return feats_per_voxel[:, :, 0, :] # [b, num , c]

    def extract_point_from_voxel(self, voxels, idxs):
        # voxels: [b, l, c]; idxs: [b, num]
        feats_per_point = batched_masked_gather(voxels, idxs[:,:,None].long(), idxs[:, :, None]>=0, fill_value=0)
        # print('point per point: ', feats_per_point.shape)
        return feats_per_point[:, :, 0, :]# [b, num, c]

    def extract_point_from_image(self, image, voxels, point_idxs):
        feats_per_voxel = self.extract_voxel_from_image(image, voxels)
        feats_per_point = self.extract_point_from_voxel(feats_per_voxel, point_idxs)
        # print('point per point merged: ', feats_per_point.shape)
        return feats_per_point

    # adapted from zeroflow
    def concat_feats(self, feats_vote, feats_before, feats_after):
        feats = torch.cat([feats_vote, feats_before, feats_after], dim=-1)
        return feats

    def _model_forward(self, points_src, points_dst):
        # assert points_src.shape==points_dst.shape

        self.timer[1][0].start("Voxelization")
        pseudoimages_src, voxel_infos_lst_src = self.embedder(points_src)
        pseudoimages_dst, voxel_infos_lst_dst = self.embedder(points_dst)
        self.timer[1][0].stop()
        
        # print('points_src:', points_src.shape)
        # print('points_dst:', points_dst.shape)  
        # print('pseudoimages_src:', pseudoimages_src.shape)
        # print('pseudoimages_dst:', pseudoimages_dst.shape)
        

        self.timer[1][1].start("Preprocessing")
        with torch.no_grad():
            point_masks_src, point_masks_dst, \
            point_voxel_idxs_src, point_voxel_idxs_dst, \
            point_offsets_src, \
            voxels_src, voxels_dst, \
            knn_idxs_src, knn_idxs_dst = self.preprocessing(points_src, points_dst, voxel_infos_lst_src, voxel_infos_lst_dst)
        self.timer[1][1].stop()
        
        self.timer[1][2].start("Feature extraction")
        pseudoimages_grid = self.backbone(pseudoimages_src, pseudoimages_dst)

        feats_point_src_init = self.extract_point_from_image(torch.cat([pseudoimages_src, pseudoimages_dst], dim=1), voxels_src, point_voxel_idxs_src)
        feats_point_src_grid = self.extract_point_from_image(pseudoimages_grid,  voxels_src, point_voxel_idxs_src)
        # print('psudoimage src:', pseudoimages_src.shape)
        # print('psudoimage dst:', pseudoimages_dst.shape)
        # print('psudoimage grid:', pseudoimages_grid.shape)
        # print('unique_voxels', voxels_src.shape)
        # print('feats_voxel_src:', feats_voxel_src.shape)
        # print('feats_voxel_dst:', feats_voxel_dst.shape)
        
        self.timer[1][2].stop()
        
        if self.using_voting:
            self.timer[1][3].start("Gathering")
            if self.use_separate_feats:
            # print('using separate features for voting')
                feats_voxel_src = self.extract_voxel_from_image(pseudoimages_src, voxels_src) # [B, N_valid_voxels, C]
                feats_voxel_dst = self.extract_voxel_from_image(pseudoimages_dst, voxels_dst) # [B, N_valid_voxels, C]
            else:
                feats_voxel_src = self.extract_voxel_from_image(pseudoimages_grid, voxels_src) # [B, N_valid_voxels, C]
                feats_voxel_dst = self.extract_voxel_from_image(pseudoimages_grid, voxels_dst) # [B, N_valid_voxels, C]
            feats_voxel_dst_inflate = batched_masked_gather(feats_voxel_dst, knn_idxs_dst.long(), knn_idxs_dst>=0, fill_value=0)
            # print('feats_voxel_dst_inflate:', feats_voxel_dst_inflate.shape)
            # print('math1:', (feats_voxel_src[:, :, None, :] * feats_voxel_dst_inflate).shape)
            # corr_src_dst = (feats_voxel_src[:, :, None, :] * feats_voxel_dst_inflate).sum(dim=-1) # [b, l, self.n]
            corr_src_dst = torch.nn.functional.cosine_similarity(feats_voxel_src[:, :, None, :], feats_voxel_dst_inflate, dim=-1)
            # print('corr_src_dst:', corr_src_dst.shape, corr_src_dst.max(-1)[0].shape, corr_src_dst.min(-1)[0].shape) # (bs, num_voxels, n)
            # print(corr_src_dst.max(-1)[0])
            # print(corr_src_dst.min(-1)[0])
            # print('corr_inflate:', corr_inflate.shape, corr_inflate[0, :10, 10:14, 100:104])
            self.timer[1][3].stop()
            
            self.timer[1][4].start("Voting")
            # print('corr_src_dst:', corr_src_dst.shape)
            voting_vols= self.vote(corr_src_dst[:, :, :, None], voxels_src, voxels_dst, knn_idxs_src, knn_idxs_dst) # [b, l, c, ny, nx]
            # print('voting  vols:', voting_vols.shape)
            
            # voting_vols_flatten = voting_vols.flatten(start_dim=-2)
            # voting_vols_max = voting_vols_flatten.max(-1)[0]
            # voting_vols_min = voting_vols_flatten.min(-1)[0]
            # print('voting  vols flatten:', voting_vols_flatten.shape, voting_vols_max.shape, voting_vols_min.shape) 
            
            # voting_vols_norm = (voting_vols - voting_vols_min[..., None, None] )/ (voting_vols_max[..., None, None] - voting_vols_min[..., None, None] + 1e-6)
            
            # print(voting_vols_norm.shape, voting_vols_norm.max(), voting_vols_norm.min())
            
            # vols_flattened = vols.view(vols.shape[0], vols.shape[1], -1)
            
            # print('vols_flattened:', vols_flattened.shape, vols_flattened[0].max(), vols_flattened[0].min())
            
            # for vol in vols_flattened:
            #     print('vol:', vol.shape, vol.max(), vol.min())
            #     topk_voting, topk_idx = torch.topk(vol, 5, dim=-1)
            #     print('topk voting:', topk_voting.shape, topk_voting.max(), topk_voting.min())
            #     print('topk idx:', topk_idx.shape)
            #     # gathered_vol = torch.gather(vol, dim =1, index = topk_idx)
            #     # print('topk idx verfi:', gathered_vol.shape)
            #     # print(torch.eq(gathered_vol, topk_voting).all())
            #     print('topk voting values:', topk_voting[:10, :])
                
            #     norm_topk_voting = topk_voting
            
            vols = self.volconv(voting_vols)
            feats_point_vol = self.extract_point_from_voxel(vols, point_voxel_idxs_src)
            # print('voting  vols after volconv:', voting_vols.shape, vols.shape, vols.max(), vols.min())
            self.timer[1][4].stop()
            

            
            # print('feats_point_vol:', feats_point_vol.shape, feats_point_vol.max(), feats_point_vol.min())
            # print('feats_point_src_init:', feats_point_src_init.shape, feats_point_src_init.max(), feats_point_src_init.min())
            # print('feats_point_src_grid:', feats_point_src_grid.shape, feats_point_src_grid.max(), feats_point_src_grid.min())
            self.timer[1][5].start("Decoding")
            if self.decoder=='simple_decoder':
                flows = self.decoder(feats_point_vol)
            else: 
                feats_cat = torch.cat([feats_point_vol, feats_point_src_init, feats_point_src_grid], -1)
                # print('feats_cat:', feats_cat.shape, feats_cat.max(), feats_cat.min())
            # print('pseudoimages_src:', pseudoimages_src.shape)
            # print('pseudoimages_dst:', pseudoimages_dst.shape)
            # print('voxels_src:', voxels_src.shape)  
            # print('point_voxel_idxs_src:', point_voxel_idxs_src.shape)
            # print('point_voxel_idxs_src:', point_voxel_idxs_src.max(), point_voxel_idxs_src.min())
            # print('feats_points_vol:', feats_point_vol.shape)
            # print('feats_points_src_init:', feats_point_src_init.shape)
            # print('feats_points_src_grid:', feats_point_src_grid.shape)
            
            # print('feats_cat:', feats_cat.shape)
            # print('pts offsets:', len(voxel_infos_lst_src))
            # for voxel_info in voxel_infos_lst_src:
            #     print('voxel_info:', voxel_info['point_offsets'].shape)
            # print('point_offsets_src:', point_offsets_src.shape)
            
                flows = self.decoder(feats_cat, point_offsets_src)
            self.timer[1][5].stop()
        else:
            feats_voxel_src = None
            feats_voxel_dst = None
            corr_src_dst = None
            voting_vols = None

            self.timer[1][5].start("Decoding")
            feats_cat = torch.cat([feats_point_src_init, feats_point_src_grid], -1)
            flows = self.decoder(feats_cat, point_offsets_src)
            self.timer[1][5].stop()
        
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
            "pseudoimages_src": pseudoimages_src,
            "pseudoimages_dst": pseudoimages_dst,
            "pseudoimages_grid": pseudoimages_grid,
            "feats_voxel_src": feats_voxel_src,
            "feats_voxel_dst": feats_voxel_dst,
            "corr": corr_src_dst,
            "voxels_src": voxels_src,
            "voting_vol": voting_vols,
            "points_src_offset": point_offsets_src,
            "points_src_voxel_idx": point_voxel_idxs_src,
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
        
        self.timer[1].start("Model Forward")
        model_res = self._model_forward(pc0s, pc1s)
        self.timer[1].stop()
        
        ret_dict = model_res
        ret_dict["pose_flow"] = pose_flows
        
        return ret_dict
        