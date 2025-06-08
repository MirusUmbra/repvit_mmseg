# Copyright (c) OpenMMLab. All rights reserved.
from pickle import NONE
import warnings

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from ..builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

# from ..builder import BACKBONES
# from ..utils import InvertedResidual, make_divisible
from ..utils import make_divisible

import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.cnn.bricks import build_activation_layer
from torch.nn import Parameter

class IBEConvModule(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, act_cfg=None, **kwargs):

        super(IBEConvModule, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.att = DifferenceAttention(out_channels, out_channels, out_channels)

        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = build_activation_layer(act_cfg) #nn.ReLU()
        self.theta = Parameter(torch.zeros([1]))

    def merge_bn(self, conv, bn):
        conv_w = conv
        conv_b = torch.zeros_like(bn.running_mean)

        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = nn.Parameter(conv_w *
                                factor.reshape([conv_w.shape[0], 1, 1, 1]))
        bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
        return weight, bias

    def forward(self, x):
        out_normal = self.conv(x)
        [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        theta = F.sigmoid(self.theta)
        outs = self.bn(out_normal) + self.bn2(out_normal - theta * out_diff)

        return self.act(outs)


class InvertedResidual(BaseModule):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor
    """

    def __init__(self,
                 conv,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 groups=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
                                 f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        layers.extend([
            conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
        ],
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


@BACKBONES.register_module()
class FemtoNet(BaseModule):
    """MobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int], optional): Output from which stages.
            Default: (1, 2, 4, 7).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride，group, ibe

    # 39*4 v3
    arch_settings = [
        [1, 16, 1, 1, 1, 0],
        [4, 24, 2, 2, 1, 1],
        [4, 32, 2, 2, 1, 1],
        [4, 64, 3, 2, 1, 1],
        [4, 96, 2, 1, 1, 1],
        [4, 160, 2, 2, 2, 0],
        [4, 320, 1, 1, 2, 1]]

    def __init__(self,
                 input_chanels=3,
                 widen_factor=1.,
                 out_indices=(1, 2, 4, 6),
                 frozen_stages=-1,
                 diff_conv=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(FemtoNet, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.widen_factor = widen_factor
        self.out_indices = out_indices
        if not set(out_indices).issubset(set(range(0, 8))):
            raise ValueError('out_indices must be a subset of range'
                             f'(0, 8). But received {out_indices}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(32 * widen_factor, 8)
        self.conv1 = ConvModule(
            in_channels=input_chanels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride, groups, ibe = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio,
                diff_conv=diff_conv,
                groups=groups,
                ibe=ibe)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio, diff_conv, groups=1, ibe=1):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if diff_conv and ibe:
                conv = IBEConvModule
            else:
                conv = ConvModule

            if i >= 1:
                stride = 1
            layers.append(
                InvertedResidual(
                    conv,
                    self.in_channels,
                    out_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    groups=groups,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
                # print(f"Feature at layer {i}: shape = {x.shape}")
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(FemtoNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

