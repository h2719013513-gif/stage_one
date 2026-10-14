# -*- coding: utf-8 -*-
"""简单卷积神经网络（SimpleCNN）。

结构（与实验要求一致：两次"卷积+ReLU+池化"，再接全连接输出 10 类）：

    输入 3x32x32
      -> 卷积块1（2 个 3x3 卷积 + BN + ReLU，最大池化）   -> 32x16x16
      -> 卷积块2（2 个 3x3 卷积 + BN + ReLU，最大池化）   -> 64x8x8
      -> 展平 -> 全连接 256 + ReLU + Dropout
      -> 全连接 10

卷积核数量不多、参数量很小，属于"简单 CNN"，方便在普通显卡甚至 CPU 上训练。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import NUM_CLASSES


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """一个卷积块：两次 3x3 卷积 + BN + ReLU，最后 2x2 最大池化。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),        # 32x32 -> 16x16
            conv_block(32, 64),       # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes: int = NUM_CLASSES) -> SimpleCNN:
    return SimpleCNN(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数个数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = build_model()
    print(net)
    print(f"\n可训练参数总数：{count_parameters(net):,}")
    dummy = torch.randn(4, 3, 32, 32)
    print(f"输入 {tuple(dummy.shape)} -> 输出 {tuple(net(dummy).shape)}")
