import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import math
class ConvBlock(nn.Module):
    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size, stride, padding, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        return x

class ConvBNBlock(nn.Module):
    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size, stride, padding, bias=True)
        self.bn = nn.BatchNorm2d(out_num_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ConvGRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.convr = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.convq = nn.Linear(input_dim+hidden_dim, hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=-1)

        z = self.sigmoid(self.convz(hx))
        r = self.sigmoid(self.convr(hx))
        rh_x = torch.cat([r*h, x], dim=-1)
        print('In ConvGRU:', rh_x.shape)
        q = self.tanh(self.convq(rh_x))

        h = (1 - z) * h + z * q
        return h

class Backbone(nn.Module):
    def __init__(self, in_num_channels, out_num_channels):
        super().__init__()
        self.conv_shared = ConvBlock(in_num_channels, out_num_channels)
        self.conv = ConvBlock(out_num_channels*2, out_num_channels)

    def forward(self, x, y) -> torch.Tensor:
        x = self.conv_shared(x)
        y = self.conv_shared(y)
        xy = self.conv(torch.cat([x,y], dim=1))
        # print('backbone: ', x.shape)
        return xy


class VolConv(nn.Module):
    def __init__(self, h, w, hidden_dim=16, dim_output=64):
        super().__init__()
        assert h%2==0
        assert w%2==0
        self.conv1 = ConvBlock(in_num_channels=1, out_num_channels=hidden_dim)
        self.conv2 = ConvBlock(in_num_channels=hidden_dim, out_num_channels=hidden_dim)
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear((h//4) * (w//4) * hidden_dim, dim_output)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, h, w = x.shape
        x = self.conv1(x.view(b*l, 1, h, w))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.linear(x.view(b*l, -1))
        x = self.relu(x)
        return x.view(b, l, -1)

class VolConvBN(nn.Module):
    def __init__(self, h, w, hidden_dim=16, dim_output=64):
        super().__init__()
        assert h%2==0
        assert w%2==0
        self.conv1 = ConvBNBlock(in_num_channels=1, out_num_channels=hidden_dim, stride=2)
        self.conv2 = ConvBNBlock(in_num_channels=hidden_dim, out_num_channels=hidden_dim, stride=2)
        self.linear = nn.Linear(math.ceil(h/4) * math.ceil(w/4) * hidden_dim, dim_output)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, h, w = x.shape
        x = self.conv1(x.view(b*l, 1, h, w))
        x = self.conv2(x)
        # print(x.shape)
        x = self.linear(x.view(b*l, -1))
        # print('after linear:', x.shape)
        x = self.relu(x)
        return x.view(b, l, -1)

class Decoder(nn.Module):
    def __init__(self, dim_input=16, dim_output=3, layer_size=1):
        super().__init__()
        # self.linear = nn.Linear(m*dim_input, dim_output)
        self.offset_encoder = nn.Linear(3, 128)
        
        filter_size=128
        decoder = nn.ModuleList()
        decoder.append(torch.nn.Linear(dim_input +128, filter_size))
        decoder.append(torch.nn.GELU())
        for _ in range(layer_size-1):
            decoder.append(torch.nn.Linear(filter_size, filter_size))
            decoder.append(torch.nn.ReLU())
        decoder.append(torch.nn.Linear(filter_size, dim_output))
        
        self.decoder = nn.Sequential(*decoder)
        print(self.decoder)
        
    def forward(self, x: torch.Tensor, pts_offsets: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        pts_offsets_feats = self.offset_encoder(pts_offsets)
        x = torch.cat([x, pts_offsets_feats], dim=-1)
        # print('decoder conv: ', x.shape)
        x = self.decoder(x)
        
        return x

class GRUDecoder(nn.Module):
    def __init__(self, pseudoimage_channels=16, dim_output=3, num_iters=4):
        super().__init__()
        # self.linear = nn.Linear(m*dim_input, dim_output)
        self.offset_encoder = nn.Linear(3, pseudoimage_channels)
    
        self.gru = ConvGRU(input_dim=pseudoimage_channels, hidden_dim=pseudoimage_channels*3)
        #self.conv1d = nn.Conv1d(pseudoimage_channels*3, pseudoimage_channels*3, 1)
        # self.linear = nn.Linear(pseudoimage_channels*3, pseudoimage_channels*3)
        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*4, pseudoimage_channels//2), nn.GELU(),
            nn.Linear(pseudoimage_channels//2, 3))
        self.num_iters = num_iters
    # def forward_single(self, 
    #                    feats,
    #                    pts_offsets: torch.Tensor) -> torch.Tensor:
        
    #     pts_offsets_feats = self.offset_encoder(pts_offsets) ## [N, 128]

    #     print(' In GRU decoder: before called gru', feats.shape, pts_offsets_feats.shape)
    #     concatenated_vectors = feats
    #     for itr in range(self.num_iters):
    #         print(' In GRU decoder: before called gru', concatenated_vectors.shape, concatenated_vectors.max(), concatenated_vectors.min())
    #         concatenated_vectors = self.gru(concatenated_vectors, pts_offsets_feats)
    #         print(' In GRU decoder: after called gru', concatenated_vectors.shape, concatenated_vectors.max(), concatenated_vectors.min())
            
    #     print(' In GRU decoder: after called gru', concatenated_vectors.shape, pts_offsets_feats.shape)
    #     flow = self.decoder(torch.cat([concatenated_vectors, pts_offsets_feats], dim=1))
    #     print(' In GRU decoder: after decoder')
    #     return flow
    
    def forward(self,
                feats_cat: torch.Tensor, 
                pts_offsets: torch.Tensor) -> torch.Tensor:
        
        # flow_results = []
        # for pts_offset, feat_cat in zip(pts_offsets,feats_cat):
        #     flow = self.forward_single(feat_cat, pts_offset)
        #     flow_results.append(flow)
        
        pts_offsets_feats = self.offset_encoder(pts_offsets) ## [N, 128]

        # print(' In GRU decoder: before called gru', feats_cat.shape, pts_offsets_feats.shape)
        concatenated_vectors = feats_cat
        for itr in range(self.num_iters):
            # print(' In GRU decoder: before called gru', concatenated_vectors.shape, concatenated_vectors.max(), concatenated_vectors.min())
            concatenated_vectors = self.gru(concatenated_vectors, pts_offsets_feats)
            # print(' In GRU decoder: after called gru', concatenated_vectors.shape, concatenated_vectors.max(), concatenated_vectors.min())
        # print(' In GRU decoder: after called gru', concatenated_vectors.shape, pts_offsets_feats.shape)
        flow = self.decoder(torch.cat([concatenated_vectors, pts_offsets_feats], dim=-1))
        
        return flow
    
class SimpleDecoder(nn.Module):
    def __init__(self, dim_input=16, dim_output=3, layer_size=1):
        super().__init__()
        # self.linear = nn.Linear(m*dim_input, dim_output)
        filter_size=128
        decoder = nn.ModuleList()
        decoder.append(torch.nn.Linear(dim_input, filter_size))
        decoder.append(torch.nn.GELU())
        for _ in range(layer_size-1):
            decoder.append(torch.nn.Linear(filter_size, filter_size))
            decoder.append(torch.nn.ReLU())
        decoder.append(torch.nn.Linear(filter_size, dim_output))
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        # print('decoder conv: ', x.shape)
        x = self.decoder(x)
        
        return x

class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.nonlinearity = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.nonlinearity(batchnorm_res)

class BilinearDecoder(nn.Module):

    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         scale_factor=self.scale_factor,
                                         mode="bilinear",
                                         align_corners=False)

class UpsampleSkip(nn.Module):

    def __init__(self, skip_channels: int, latent_channels: int,
                 out_channels: int):
        super().__init__()
        self.u1_u2 = nn.Sequential(
            nn.Conv2d(skip_channels, latent_channels, 1, 1, 0),
            BilinearDecoder(2))
        self.u3 = nn.Conv2d(latent_channels, latent_channels, 1, 1, 0)
        self.u4_u5 = nn.Sequential(
            nn.Conv2d(2 * latent_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        u2_res = self.u1_u2(a)
        u3_res = self.u3(b)
        u5_res = self.u4_u5(torch.cat([u2_res, u3_res], dim=1))
        return u5_res

class FastFlowUNet(nn.Module):
    """
    Standard UNet with a few modifications:
     - Uses Bilinear interpolation instead of transposed convolutions
    """

    def __init__(self, in_num_channels, out_num_channels) -> None:
        super().__init__()

        self.encoder_step_1 = nn.Sequential(ConvWithNorms(in_num_channels, 64, 3, 2, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1))
        self.encoder_step_2 = nn.Sequential(ConvWithNorms(64, 128, 3, 2, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1))
        self.encoder_step_3 = nn.Sequential(ConvWithNorms(128, 256, 3, 2, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1))
        self.decoder_step1 = UpsampleSkip(512, 256, 256)
        self.decoder_step2 = UpsampleSkip(256, 128, 128)
        self.decoder_step3 = UpsampleSkip(128, 64, 64)
        self.decoder_step4 = nn.Conv2d(64, out_num_channels, 3, 1, 1)

    def forward(self, pc0_B: torch.Tensor,
                pc1_B: torch.Tensor) -> torch.Tensor:

        expected_channels = 32
        assert pc0_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc0_B.shape[1]}"
        assert pc1_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc1_B.shape[1]}"

        pc0_F = self.encoder_step_1(pc0_B)
        pc0_L = self.encoder_step_2(pc0_F)
        pc0_R = self.encoder_step_3(pc0_L)

        pc1_F = self.encoder_step_1(pc1_B)
        pc1_L = self.encoder_step_2(pc1_F)
        pc1_R = self.encoder_step_3(pc1_L)

        Rstar = torch.cat([pc0_R, pc1_R],
                          dim=1)  # torch.Size([1, 512, 64, 64])
        Lstar = torch.cat([pc0_L, pc1_L],
                          dim=1)  # torch.Size([1, 256, 128, 128])
        Fstar = torch.cat([pc0_F, pc1_F],
                          dim=1)  # torch.Size([1, 128, 256, 256])
        Bstar = torch.cat([pc0_B, pc1_B],
                          dim=1)  # torch.Size([1, 64, 512, 512])

        S = self.decoder_step1(Rstar, Lstar)
        T = self.decoder_step2(S, Fstar)
        U = self.decoder_step3(T, Bstar)
        V = self.decoder_step4(U)

        return V