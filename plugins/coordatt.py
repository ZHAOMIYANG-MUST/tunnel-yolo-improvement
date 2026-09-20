"""Coordinate Attention（坐标注意力，Hou et al., CVPR 2021）。

设计动机（来自本数据集的数据分析，见 analysis/数据分析报告.md）：
    统计 2510 个渗漏框发现，23% 的框高度接近 1.0（纵向贯穿整个画面），
    高中位数 0.75，38.9% 的框长宽比处于极端区间——渗漏在形态上表现为
    "沿隧道纵向延伸的细长条"，而非圆形小斑点。

    普通通道注意力（如 SE/CBAM 的通道分支）把整张特征图全局池化成
    1×1，会丢失空间位置信息；CBAM 的空间分支用 7×7 卷积，感受野是
    各向同性的方形，对"又长又窄"的目标不敏感。

    CA 的做法是把特征沿高度和宽度**两个方向分别做一维池化**，得到两组
    带方向信息的位置编码，再各自生成方向感知的注意力权重。因此横向/纵向
    的细长结构会被显式建模——正好对应本数据集"纵向长条"的形态特点。

参考文献：Hou Q, Zhou D, Feng J. Coordinate Attention for Efficient Mobile
    Network Design. CVPR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    """h-swish 激活（原论文使用），x * relu6(x + 3) / 6。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=False) / 6.0


class CoordAtt(nn.Module):
    """坐标注意力模块，输入输出通道数一致，可直接插在任意特征图之后。

    Args:
        c1: 输入通道数
        c2: 输出通道数（默认与输入相同；本项目中始终保持相同）
        reduction: 中间层压缩比（原论文默认 32）
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 32):
        super().__init__()
        c2 = c2 or c1
        # 中间通道数下限 8，避免小通道层（如 16、32）被压得过窄
        mip = max(8, c1 // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿宽度压缩 → (n,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿高度压缩 → (n,c,1,w)

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()

        # 两个方向各自生成注意力权重
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # 1) 两个方向的一维池化（保留各自方向的位置信息）
        x_h = self.pool_h(x)  # (n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n, c, w, 1)

        # 2) 拼接后共享变换，压缩通道
        y = torch.cat([x_h, x_w], dim=2)  # (n, c, h + w, 1)
        y = self.act(self.bn1(self.conv1(y)))

        # 3) 拆回两个方向，各自生成注意力图
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (n, c, 1, w)

        a_h = self.conv_h(x_h).sigmoid()  # (n, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (n, c, 1, w)

        # 4) 两个方向权重相乘后再加权原特征
        return identity * a_h * a_w
