from functools import partial
import torch
from torch import nn as nn
from torch.nn import functional as F
import importlib


class AttentionLayer3D(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(AttentionLayer3D, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class ChannelSELayer3D(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):

        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(
            self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(
            batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):

    def __init__(self, num_channels):

        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None):

        # channel squeeze
        batch_size, channel, D, H, W = x.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(
            x, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):

        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(
            self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class CRU(nn.Module):

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1/2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha*op_channel)
        self.low_channel = low_channel = op_channel-up_channel
        self.squeeze1 = nn.Conv3d(
            up_channel, up_channel//squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv3d(
            low_channel, low_channel//squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv3d(up_channel//squeeze_radio, op_channel, kernel_size=group_kernel_size,
                             stride=1, padding=group_kernel_size//2, groups=group_size)
        self.PWC1 = nn.Conv3d(up_channel//squeeze_radio,
                              op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv3d(low_channel//squeeze_radio, op_channel -
                              low_channel//squeeze_radio, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1)//2, dim=1)
        return out1+out2


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight/sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(
            reweigts), reweigts)
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(
            reweigts), reweigts)
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([x_11+x_22, x_12+x_21], dim=1)


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1/2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class SingleConv(nn.Sequential):
    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        dropout_prob (float): dropout probability, default 0.1
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8,
                 padding=1, dropout_prob=0.1, is3d=True):
        super(SingleConv, self).__init__()

        for name, module in self.create_conv(in_channels, out_channels, kernel_size, order,
                                             num_groups, padding, dropout_prob, is3d):
            self.add_module(name, module)

    def create_conv(self, in_channels, out_channels, kernel_size, order, num_groups, padding,
                    dropout_prob, is3d):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size(int or tuple): size of the convolving kernel
            order (string): order of things, e.g.
                'cr' -> conv + ReLU
                'gcr' -> groupnorm + conv + ReLU
                'cl' -> conv + LeakyReLU
                'ce' -> conv + ELU
                'bcr' -> batchnorm + conv + ReLU
                'cbrd' -> conv + batchnorm + ReLU + dropout
                'cbrD' -> conv + batchnorm + ReLU + dropout2d
            num_groups (int): number of groups for the GroupNorm
            padding (int or tuple): add zero-padding added to all three sides of the input
            dropout_prob (float): dropout probability
            is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
        """
        assert 'c' in order, "Conv layer MUST be present"
        assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

        modules = []
        for i, char in enumerate(order):
            if char == 'r':
                modules.append(('ReLU', nn.ReLU(inplace=True)))
            elif char == 'l':
                modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
            elif char == 'e':
                modules.append(('ELU', nn.ELU(inplace=True)))
            elif char == 'c':
                # add learnable bias only in the absence of batchnorm/groupnorm
                bias = not ('g' in order or 'b' in order)
                if is3d:
                    conv = nn.Conv3d(in_channels, out_channels,
                                     kernel_size, padding=padding, bias=bias)
                else:
                    conv = nn.Conv2d(in_channels, out_channels,
                                     kernel_size, padding=padding, bias=bias)

                modules.append(('conv', conv))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                if is_before_conv:
                    num_channels = in_channels
                else:
                    num_channels = out_channels

                # use only one group if the given number of groups is greater than the number of channels
                if num_channels < num_groups:
                    num_groups = 1

                assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
                modules.append(('groupnorm', nn.GroupNorm(
                    num_groups=num_groups, num_channels=num_channels)))
            elif char == 'b':
                is_before_conv = i < order.index('c')
                if is3d:
                    bn = nn.BatchNorm3d
                else:
                    bn = nn.BatchNorm2d

                if is_before_conv:
                    modules.append(('batchnorm', bn(in_channels)))
                else:
                    modules.append(('batchnorm', bn(out_channels)))
            elif char == 'd':
                modules.append(('dropout', nn.Dropout3d(p=dropout_prob)))
            elif char == 'D':
                modules.append(('dropout2d', nn.Dropout2d(p=dropout_prob)))
            else:
                raise ValueError(
                    f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'd', 'D']")

        return modules

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr',
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            if upscale == 1:
                conv1_out_channels = out_channels
            else:
                conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # check if dropout_prob is a tuple and if so
        # split it for different dropout probabilities for each convolution.
        if isinstance(dropout_prob, list) or isinstance(dropout_prob, tuple):
            dropout_prob1 = dropout_prob[0]
            dropout_prob2 = dropout_prob[1]
        else:
            dropout_prob1 = dropout_prob2 = dropout_prob

        self.conv1_in_channels, self.conv1_out_channels = conv1_in_channels, conv1_out_channels
        self.conv2_in_channels, self.conv2_out_channels = conv2_in_channels, conv2_out_channels

        self.dropout_prob1 = dropout_prob1
        self.dropout_prob2 = dropout_prob2

        if encoder:

            encoder_conv_list = []
            for i in range(1, 1+repeats):
                encoder_conv_list.append(SingleConv(self.conv1_in_channels, self.conv1_out_channels, kernel_size, order, num_groups,
                                                    padding=padding, dropout_prob=self.dropout_prob1, is3d=is3d))

            self.conv1 = nn.Sequential(*encoder_conv_list)

            self.conv2 = nn.Sequential(SingleConv(self.conv2_in_channels, self.conv2_out_channels, kernel_size, order, num_groups,
                                       padding=padding, dropout_prob=self.dropout_prob2, is3d=is3d))

        else:

            self.conv1 = nn.Sequential(SingleConv(self.conv1_in_channels, self.conv1_out_channels, kernel_size, order, num_groups,
                                       padding=padding, dropout_prob=self.dropout_prob1, is3d=is3d))
            deconder_conv_list = []
            for i in range(1, 1+repeats):
                deconder_conv_list.append(SingleConv(self.conv2_in_channels, self.conv2_out_channels, kernel_size, order, num_groups,
                                                     padding=padding, dropout_prob=self.dropout_prob2, is3d=is3d))

            self.conv2 = nn.Sequential(*deconder_conv_list)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class DoubleConvSE(DoubleConv):
    def __init__(self, in_channels, out_channels, encoder, se_module="scse", kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(
                num_channels=self.conv2_out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(
                num_channels=self.conv2_out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(
                num_channels=self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.se_module(x)

        return x

class DoubleConvSC(DoubleConv):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        self.sc_model = ScConv(self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.sc_model(x)

        return x


class DoubleConvResidual(DoubleConv):

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)
        
            
        if self.conv1_in_channels == self.conv2_out_channels:
            self.residual_edge = nn.Identity()
        else:
            self.residual_edge = nn.Sequential(nn.Conv3d(
                    self.conv1_in_channels,
                    self.conv2_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),nn.BatchNorm3d(self.conv2_out_channels))

        if 'l' in order:
            self.non_linearity=nn.LeakyReLU(negative_slope=0.1, inplace=True)
            
        elif 'e' in order:
            self.non_linearity=nn.ELU(inplace=True)

        else:
            self.non_linearity=nn.ReLU(inplace=True)
            
        self.residual_edge.add_module(name="non_linearity", module=self.non_linearity)

        if 'd' in order:
            self.residual_edge.add_module(name="dropout", module=nn.Dropout3d(p=dropout_prob))
                

    def forward(self, x):

        residual = self.residual_edge(x)

        out = self.conv1(x)
        out = self.conv2(out)

        return self.non_linearity(out + residual)

class DoubleConvResidualSE(DoubleConvResidual):

    def __init__(self, in_channels, out_channels, encoder, se_module="scse", kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(
                num_channels=self.conv2_out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(
                num_channels=self.conv2_out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(
                num_channels=self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.se_module(x)

        return x


class DoubleConvResidualSC(DoubleConvResidual):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        self.sc_model = ScConv(self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.sc_model(x)

        return x


class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, is3d=True, repeats=1, **kwargs):
        super(ResNetBlock, self).__init__()

        if in_channels != out_channels:
            # conv1x1 for increasing the number of channels
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,
                                is3d=is3d)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups, is3d=is3d)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution to bring the number of channels to out_channels
        residual = self.conv1(x)

        # residual block
        out = self.conv2(residual)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class ResNetBlockSE(ResNetBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, se_module='scse', repeats=1, **kwargs):
        super(ResNetBlockSE, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, order=order,
            num_groups=num_groups, **kwargs)
        assert se_module in ['scse', 'cse', 'sse']
        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(
                num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(
                num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(num_channels=out_channels)

    def forward(self, x):
        out = super().forward(x)
        out = self.se_module(out)
        return out


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True, repeats=1,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.1, is3d=True):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         repeats=repeats,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         upscale=upscale,
                                         dropout_prob=dropout_prob,
                                         is3d=is3d)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=2, basic_module=DoubleConv, repeats=1,
                 conv_layer_order='gcr', num_groups=8, padding=1, upsample='default', use_attn=False,
                 dropout_prob=0.1, is3d=True):
        super(Decoder, self).__init__()

        # perform concat joining per default
        concat = True

        # don't adapt channels after join operation
        adapt_channels = False

        self.attention = None

        if upsample is not None and upsample != 'none':
            if upsample == 'default':

                if basic_module == ResNetBlock or basic_module == ResNetBlockSE:
                    upsample = 'deconv'  # use deconvolution upsampling
                    concat = False  # use summation joining
                    adapt_channels = True  # adapt channels after joining
                else:
                    upsample = 'nearest'  # use nearest neighbor interpolation for upsampling
                    concat = True  # use concat joining
                    adapt_channels = False  # don't adapt channels

            # perform deconvolution upsampling if mode is deconv
            if upsample == 'deconv':
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor,
                                                          is3d=is3d)
            else:
                self.upsampling = InterpolateUpsampling(mode=upsample)
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        # perform joining operation
        self.joining = partial(self._joining, concat=concat)

        # adapt the number of in_channels for the ResNetBlock
        if adapt_channels is True:
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         repeats=repeats,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         dropout_prob=dropout_prob,
                                         is3d=is3d)

        if use_attn:
            if not adapt_channels:
                self.attention = AttentionLayer3D(
                    out_channels, in_channels - out_channels, out_channels)
            else:
                self.attention = AttentionLayer3D(
                    in_channels, in_channels, in_channels)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)

        if self.attention is not None:
            encoder_features = self.attention(g=x, x=encoder_features)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):

    class Upsample(nn.Module):

        def __init__(self, conv_transposed, is3d):
            super().__init__()
            self.conv_transposed = conv_transposed
            self.is3d = is3d

        def forward(self, x, size):
            x = self.conv_transposed(x)
            if self.is3d:
                output_size = x.size()[-3:]
            else:
                output_size = x.size()[-2:]
            if output_size != size:
                return F.interpolate(x, size=size)
            return x

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, is3d=True):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        if is3d is True:
            conv_transposed = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)
        else:
            conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)
        upsample = self.Upsample(conv_transposed, is3d)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


class AbstractUNet(nn.Module):

    def __init__(self, in_channels, out_channels, basic_module, final_activation='sigmoid', f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_kernel_size=3, pool_type='avg', pool_kernel_size=2, repeats=1, use_attn=False,
                 conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, is3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = self.number_of_features_per_level(
                f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders, self.encoders_maps = self.create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
                                             conv_padding, conv_upscale, dropout_prob,
                                             layer_order, num_groups, pool_kernel_size, is3d, repeats,pool_type)

        # create decoder path
        self.decoders, self.decoders_maps = self.create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding,
                                             layer_order, num_groups, upsample, dropout_prob,
                                             is3d, repeats, use_attn)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        self.final_activation = self.activate(final_activation)

    def activate(self, activation):

        if activation is not None:
            if activation == 'relu':
                return nn.ReLU(inplace=True)
            elif activation == 'leakyrelu':
                return nn.LeakyReLU(inplace=True)
            elif activation == 'prelu':
                return nn.PReLU()
            elif activation == 'celu':
                return nn.CELU()
            elif activation == 'sigmoid':
                return nn.Sigmoid()
            elif activation == 'softmax':
                return nn.Softmax(dim=1)
            elif activation == 'tanh':
                return nn.Tanh()
            elif activation == 'softsign':
                return nn.Softsign()
            elif activation == 'hardtanh':
                return nn.Hardtanh(min_val=0.0, max_val=1.0)
            else:
                raise NotImplementedError(
                    'Option {} not implemented. Available options: relu | leakyrelu | prelu | celu | sigmoid | softmax ;'.format(activation))
        else:
            return None

    def number_of_features_per_level(self, init_channel_number, num_levels):
        return [init_channel_number * 2 ** k for k in range(num_levels)]

    def create_encoders(self, in_channels, f_maps, basic_module, conv_kernel_size, conv_padding,
                        conv_upscale, dropout_prob,
                        layer_order, num_groups, pool_kernel_size, is3d, repeats, pool_type):
        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        out_feature_list = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                # apply conv_coord only in the first encoder if any
                encoder = Encoder(in_channels, out_feature_num,
                                  apply_pooling=False,  # skip pooling in the firs encoder
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding,
                                  upscale=conv_upscale,
                                  dropout_prob=dropout_prob,
                                  is3d=is3d,
                                  repeats=1,
                                  pool_type=pool_type)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding,
                                  upscale=conv_upscale,
                                  dropout_prob=dropout_prob,
                                  is3d=is3d,
                                  repeats=repeats,
                                  pool_type=pool_type)

            encoders.append(encoder)
            out_feature_list.append(out_feature_num)
            

        return nn.ModuleList(encoders), out_feature_list

    def create_decoders(self, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                        num_groups, upsample, dropout_prob, is3d, repeats, use_attn):
        # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
        decoders = []
        out_feature_list = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if not (basic_module == ResNetBlock or basic_module == ResNetBlockSE) and upsample != 'deconv':
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]

            decoder = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding,
                              upsample=upsample,
                              dropout_prob=dropout_prob,
                              is3d=is3d,
                              repeats=repeats,
                              use_attn=use_attn,
                              )
            decoders.append(decoder)
            out_feature_list.append(out_feature_num)
        return nn.ModuleList(decoders), out_feature_list

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, pool_type='max', use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     repeats=repeats,
                                     use_attn=use_attn,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     final_activation=final_activation,
                                     is3d=True,
                                     pool_type=pool_type)


class UNet3D_SE(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_SE, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        repeats=repeats,
                                        use_attn=use_attn,
                                        basic_module=DoubleConvSE,
                                        f_maps=f_maps,
                                        layer_order=layer_order,
                                        num_groups=num_groups,
                                        num_levels=num_levels,
                                        conv_kernel_size=conv_kernel_size,
                                        conv_padding=conv_padding,
                                        conv_upscale=conv_upscale,
                                        upsample=upsample,
                                        dropout_prob=dropout_prob,
                                        final_activation=final_activation,
                                        is3d=True)


class UNet3D_SC(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_SC, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        repeats=repeats,
                                        use_attn=use_attn,
                                        basic_module=DoubleConvSC,
                                        f_maps=f_maps,
                                        layer_order=layer_order,
                                        num_groups=num_groups,
                                        num_levels=num_levels,
                                        conv_kernel_size=conv_kernel_size,
                                        conv_padding=conv_padding,
                                        conv_upscale=conv_upscale,
                                        upsample=upsample,
                                        dropout_prob=dropout_prob,
                                        final_activation=final_activation,
                                        is3d=True)
        

class UNet3D_Residual(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_Residual, self).__init__(in_channels=in_channels,
                                              out_channels=out_channels,
                                              repeats=repeats,
                                              use_attn=use_attn,
                                              basic_module=DoubleConvResidual,
                                              f_maps=f_maps,
                                              layer_order=layer_order,
                                              num_groups=num_groups,
                                              num_levels=num_levels,
                                              conv_kernel_size=conv_kernel_size,
                                              conv_padding=conv_padding,
                                              conv_upscale=conv_upscale,
                                              upsample=upsample,
                                              dropout_prob=dropout_prob,
                                              final_activation=final_activation,
                                              is3d=True)


class UNet3D_ResidualSE(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_ResidualSE, self).__init__(in_channels=in_channels,
                                                out_channels=out_channels,
                                                repeats=repeats,
                                                use_attn=use_attn,
                                                basic_module=DoubleConvResidualSE,
                                                f_maps=f_maps,
                                                layer_order=layer_order,
                                                num_groups=num_groups,
                                                num_levels=num_levels,
                                                conv_kernel_size=conv_kernel_size,
                                                conv_padding=conv_padding,
                                                conv_upscale=conv_upscale,
                                                upsample=upsample,
                                                dropout_prob=dropout_prob,
                                                final_activation=final_activation,
                                                is3d=True)


class UNet3D_ResidualSC(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_ResidualSC, self).__init__(in_channels=in_channels,
                                                out_channels=out_channels,
                                                repeats=repeats,
                                                use_attn=use_attn,
                                                basic_module=DoubleConvResidualSC,
                                                f_maps=f_maps,
                                                layer_order=layer_order,
                                                num_groups=num_groups,
                                                num_levels=num_levels,
                                                conv_kernel_size=conv_kernel_size,
                                                conv_padding=conv_padding,
                                                conv_upscale=conv_upscale,
                                                upsample=upsample,
                                                dropout_prob=dropout_prob,
                                                final_activation=final_activation,
                                                is3d=True)


class ResidualUNet3D(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', use_attn=False,
                 num_groups=8, num_levels=5, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             use_attn=use_attn,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             final_activation=final_activation,
                                             is3d=True)


class ResidualUNetSE3D(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', use_attn=False,
                 num_groups=8, num_levels=5, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               use_attn=use_attn,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               conv_padding=conv_padding,
                                               conv_upscale=conv_upscale,
                                               final_activation=final_activation,
                                               upsample=upsample,
                                               dropout_prob=dropout_prob,
                                               is3d=True)


if __name__ == "__main__":
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    for i in range(10):
        torch.cuda.empty_cache()
        image = torch.randn(1, 2, 112, 112, 112).cuda()
        model = UNet3D_Residual(in_channels=2,
                                out_channels=1,
                                f_maps=32,
                                layer_order="cbrd",
                                dropout_prob=0.25,
                                repeats=1,
                                final_activation="sigmoid",
                                conv_kernel_size=3,
                                conv_padding=1,
                                use_attn=True,
                                num_levels=5).cuda()

        print(model.__class__.__name__)
        #print(model)
        out = model(image)
        print(out.shape)
