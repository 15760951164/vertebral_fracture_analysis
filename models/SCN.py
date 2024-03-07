import torch
import torch.nn as nn
from typing import Sequence
from models.UNet import UNet3D_Residual, UNet3D_ResidualSE, UNet3D_SE, UNet3D
# from UNet import UNet3D_Residual, UNet3D_ResidualSE, UNet3D_SE, UNet3D


class SCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_maps: int = 32,
        dropout_prob: float = 0.25
    ):
        super().__init__()

        self.scnet_local = UNet3D(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps, conv_kernel_size=3, conv_padding=1,
                                  num_levels=5, layer_order="cbld", pool_type='max',dropout_prob=dropout_prob,  repeats=1, final_activation="tanh", use_attn=False)

        self.local_heatmaps = nn.Identity()

        self.down = nn.MaxPool3d(2, 2, ceil_mode=True)

        self.scnet_spatial = UNet3D(in_channels=out_channels, out_channels=out_channels, f_maps=f_maps, conv_kernel_size=3, conv_padding=1,
                                    num_levels=5, layer_order="cbld", pool_type='max',dropout_prob=dropout_prob, repeats=0, final_activation=None, use_attn=False)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Tanh()
        )

        self.spatial_heatmaps = nn.Identity()

    def forward(self, inputs):

        node = self.scnet_local(inputs)

        local_heatmaps = node = self.local_heatmaps(node)

        node = self.down(node)

        node = self.scnet_spatial(node)

        node = self.spatial_heatmaps(node)

        spatial_heatmaps = self.up(node)

        heatmaps = local_heatmaps * spatial_heatmaps

        return heatmaps


if __name__ == "__main__":
    image = torch.randn(1, 1, 96, 96, 128)
    model = SCN(in_channels=1, out_channels=25, f_maps=32)
    print(model.__class__.__name__)
    print(model)
    out = model(image)
    print(out.shape)
