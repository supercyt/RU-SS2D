"""
Resblock is much strong than normal conv

Provide api for multiscale intermeidate fuion
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock

DEBUG = False

class BiFPNLayer(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_channels)

        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.conv5 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_channels)

    def forward(self, inputs):
        # inputs: list of feature maps with shape [B, C, H, W]
        # Assume input order: [P2, P4, P8] (from high to low resolution)

        P2, P4, P8 = inputs

        # Top-down
        w = F.relu(self.w1)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        P4_td = self.bn4(self.conv4(weight[0] * P4 + weight[1] * F.interpolate(P8, size=P4.shape[2:], mode='nearest')))
        P2_td = self.bn3(self.conv3(weight[0] * P2 + weight[1] * F.interpolate(P4_td, size=P2.shape[2:], mode='nearest')))

        # Bottom-up
        w2 = F.relu(self.w2)
        weight2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

        P4_out = self.bn4(self.conv4(
            weight2[0] * P4 +
            weight2[1] * F.max_pool2d(P2_td, kernel_size=2) +
            weight2[2] * P4_td
        ))

        P8_out = self.bn5(self.conv5(
            weight2[0] * P8 +
            weight2[1] * F.max_pool2d(P4_out, kernel_size=2) +
            weight2[2] * P8
        ))

        return [P2_td, P4_out, P8_out]


class ResNetBEVBackboneWithBiFPN(nn.Module):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg['layer_nums']
        layer_strides = self.model_cfg['layer_strides']
        num_filters = self.model_cfg['num_filters']
        self.num_levels = len(layer_nums)

        self.resnet = ResNetModified(BasicBlock,
                                     layer_nums,
                                     layer_strides,
                                     num_filters,
                                     inplanes=model_cfg.get('inplanes', 64))

        # 统一通道数用于 BiFPN
        bifpn_channels = model_cfg.get('bifpn_channels', 128)
        self.lateral_convs = nn.ModuleList()
        for nf in num_filters:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(nf, bifpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(bifpn_channels),
                nn.ReLU(inplace=True)
            ))

        self.bifpn = BiFPNLayer(num_channels=bifpn_channels)

        self.output_conv = nn.Sequential(
            nn.Conv2d(bifpn_channels * self.num_levels, bifpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bifpn_channels),
            nn.ReLU(inplace=True)
        )

        self.num_bev_features = bifpn_channels

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        # 提取多尺度特征
        features = self.resnet(spatial_features)
        features = [lateral_conv(f) for f, lateral_conv in zip(features, self.lateral_convs)]

        # BiFPN 融合
        fused_features = self.bifpn(features)

        # 上采样至统一大小并拼接
        target_size = fused_features[0].shape[2:]
        upsampled = [F.interpolate(f, size=target_size, mode='nearest') for f in fused_features]
        x = torch.cat(upsampled, dim=1)

        x = self.output_conv(x)

        data_dict['spatial_features_2d'] = x
        return data_dict

    