import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super(DepthwiseSeparableBlock, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels
        )
        self.pointwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

        #self.activation = nn.ReLU()  # Optional but recommended

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x #self.activation(x)


class Model(nn.Module):
    """
    Implements num_blocks of depthwise separable convolutions with kernel size 5 and a linear prediction head
    """

    def __init__(self, configs, num_blocks=3):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.blocks = nn.Sequential(
            *[DepthwiseSeparableBlock(self.enc_in) for _ in range(num_blocks)]
        )

        self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Sequence Length, Channels] -> [Batch, Channels, Sequence Length]
        x = x.permute(0, 2, 1)

        x = self.blocks(x)
        x = self.linear(x)

        # back to [Batch, Sequence Length, Channels]
        x = x.permute(0, 2, 1)
        return x