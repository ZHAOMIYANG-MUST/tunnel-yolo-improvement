"""CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)。

串联两层注意力：
  1) 通道注意力——对特征图做全局平均池化与全局最大池化，经共享 MLP 后相加、Sigmoid，
     回答"哪些通道重要"；
  2) 空间注意力——在通道维取平均与最大值并拼接，经 7x7 卷积、Sigmoid，
     回答"图像上哪个位置重要"。

用于改进实验②：隧道内低光照、水渍与衬砌背景对比度低，CBAM 让网络把响应集中到病害区域，
抑制电缆、接缝等背景纹理造成的误检。输入输出同形状，属即插即用模块。

注册方式见 plugins/__init__.py。
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力子模块。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """空间注意力子模块。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size 只能是 3 或 7")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        return self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """通道注意力 + 空间注意力串行堆叠。

    在模型 yaml 中写作 ``[-1, 1, CBAM, [c]]``，c 为当前特征图通道数。
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)
