import torch
import torch.nn as nn

from typing import Tuple
from . import ConvWithNorms


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


class FastFlow3DUNet(nn.Module):
    """
    Standard UNet with a few modifications:
     - Uses Bilinear interpolation instead of transposed convolutions
    
    Note by Shiming:
    - modified to the flexible number of input and output channels
    """

    def __init__(self, input_channels=32, output_channels=64) -> None:
        super().__init__()
        
        c0 = input_channels
        c1 = c0 * 2 # 64
        c2 = c0 * 4 # 128
        c3 = c0 * 8 # 256
        c4 = c0 * 16 # 512
        
        self.encoder_step_1 = nn.Sequential(ConvWithNorms(c0, c1, 3, 2, 1),
                                            ConvWithNorms(c1, c1, 3, 1, 1),
                                            ConvWithNorms(c1, c1, 3, 1, 1),
                                            ConvWithNorms(c1, c1, 3, 1, 1))
        self.encoder_step_2 = nn.Sequential(ConvWithNorms(c1, c2, 3, 2, 1),
                                            ConvWithNorms(c2, c2, 3, 1, 1),
                                            ConvWithNorms(c2, c2, 3, 1, 1),
                                            ConvWithNorms(c2, c2, 3, 1, 1),
                                            ConvWithNorms(c2, c2, 3, 1, 1),
                                            ConvWithNorms(c2, c2, 3, 1, 1))
        self.encoder_step_3 = nn.Sequential(ConvWithNorms(c2, c3, 3, 2, 1),
                                            ConvWithNorms(c3, c3, 3, 1, 1),
                                            ConvWithNorms(c3, c3, 3, 1, 1),
                                            ConvWithNorms(c3, c3, 3, 1, 1),
                                            ConvWithNorms(c3, c3, 3, 1, 1),
                                            ConvWithNorms(c3, c3, 3, 1, 1))
        self.decoder_step1 = UpsampleSkip(c4, c3, c3)
        self.decoder_step2 = UpsampleSkip(c3, c2, c2)
        self.decoder_step3 = UpsampleSkip(c2, c1, c1)
        self.decoder_step4 = nn.Conv2d(c1, output_channels, 3, 1, 1)

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