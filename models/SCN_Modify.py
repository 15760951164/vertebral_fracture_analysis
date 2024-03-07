import torch
import torch.nn as nn
from typing import Sequence
from models.UNet import UNet3D_Residual, UNet3D_ResidualSE, UNet3D_SE, UNet3D
#from UNet import UNet3D_Residual, UNet3D_ResidualSE, UNet3D_SE, UNet3D


class LocalNetwork(UNet3D):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, pool_type='max', use_attn=False, num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid", conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super().__init__(in_channels, out_channels, f_maps, layer_order, repeats, pool_type, use_attn, num_groups,
                         num_levels, conv_kernel_size, conv_padding, final_activation, conv_upscale, upsample, dropout_prob, **kwargs)

    def forward(self, x):
        # encoder
        encoders_features = []
        decoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        # decoder
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
            decoders_features.append(x)

        x = self.final_conv(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x, encoders_features, decoders_features

class GlobalNetwork(UNet3D):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, pool_type='max', use_attn=False, num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid", conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super().__init__(in_channels, out_channels, f_maps, layer_order, repeats, pool_type, use_attn, num_groups,
                         num_levels, conv_kernel_size, conv_padding, final_activation, conv_upscale, upsample, dropout_prob, **kwargs)

        self.down = nn.MaxPool3d(2, 2, ceil_mode=True)

        encoders_heatmap = []
        decoders_heatmap = []
        for en_maps, de_maps in zip(self.encoders_maps[:4], reversed(self.decoders_maps)):
            encoders_heatmap.append(nn.Sequential(
                nn.Conv3d(en_maps*2, en_maps, 3, 1, 1, bias=False),
                nn.BatchNorm3d(en_maps)))

            decoders_heatmap.append(nn.Sequential(
                nn.Conv3d(de_maps*2, de_maps, 3, 1, 1, bias=False),
                nn.BatchNorm3d(de_maps)))

        self.encoders_heatmap = nn.ModuleList(encoders_heatmap)
        self.decoders_heatmap = nn.ModuleList(decoders_heatmap)
        
        if 'l' in layer_order:
            for e_layer, d_layer in zip(self.encoders_heatmap, self.decoders_heatmap):
                e_layer.add_module("non_linearity", nn.LeakyReLU(negative_slope=0.1, inplace=True))
                d_layer.add_module("non_linearity", nn.LeakyReLU(negative_slope=0.1, inplace=True))
                
        if 'r' in layer_order:
            for e_layer, d_layer in zip(self.encoders_heatmap, self.decoders_heatmap):
                e_layer.add_module("non_linearity", nn.ReLU(inplace=True))
                d_layer.add_module("non_linearity", nn.ReLU(inplace=True))
                
        if 'd' in layer_order:
            for e_layer, d_layer in zip(self.encoders_heatmap, self.decoders_heatmap):
                e_layer.add_module("dropout", nn.Dropout3d(dropout_prob))
                d_layer.add_module("dropout", nn.Dropout3d(dropout_prob))

    def forward(self, x, last_encoder_features, last_decoder_features):
        # encoder
        encoders_features = []
        for idx, encoder in enumerate(self.encoders):
            if idx > 0:
                x_hat = torch.cat([x, self.down(last_encoder_features[4-idx])], dim=1)
                x = self.encoders_heatmap[idx-1](x_hat)
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        # decoder
        for idx, (decoder, encoder_features) in enumerate(zip(self.decoders, encoders_features)):

            x = decoder(encoder_features, x)
            x_hat = torch.cat([x, self.down(last_decoder_features[idx])], dim=1)
            x = self.decoders_heatmap[3-idx](x_hat)

        x = self.final_conv(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

class SCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_maps: int = 64,
        dropout_prob: float = 0.1
    ):
        super().__init__()

        self.scnet_local = LocalNetwork(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps, conv_kernel_size=3, conv_padding=1,
                                        num_levels=5, layer_order="cbld", dropout_prob=dropout_prob, repeats=1, final_activation="tanh", use_attn=False)

        self.local_heatmaps = nn.Identity()

        self.down = nn.MaxPool3d(2, 2, ceil_mode=True)

        self.scnet_spatial = GlobalNetwork(in_channels=out_channels, out_channels=out_channels, f_maps=f_maps, conv_kernel_size=3, conv_padding=1,
                                           num_levels=5, layer_order="cbld", dropout_prob=dropout_prob, repeats=0, final_activation=None, use_attn=False)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Tanh()
        )

        self.spatial_heatmaps = nn.Identity()
    
    def forward(self, inputs):

        node, encoders_features, decoders_features = self.scnet_local(inputs)

        local_heatmaps = node = self.local_heatmaps(node)

        node = self.down(node)

        node = self.scnet_spatial(node, encoders_features, decoders_features)

        node = self.up(node)

        spatial_heatmaps = node = self.spatial_heatmaps(node)

        heatmaps = local_heatmaps * spatial_heatmaps 
        
        return heatmaps

if __name__ == "__main__":
    for i in range(10):
        image = torch.randn(1, 1, 96, 96, 128).cuda()
        model = SCN(in_channels=1, out_channels=1, f_maps=32).cuda()
        print(model.__class__.__name__)
        #print(model)
        out = model(image)
        print(out.shape)
