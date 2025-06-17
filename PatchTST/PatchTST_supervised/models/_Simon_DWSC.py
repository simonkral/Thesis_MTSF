import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(DepthwiseSeparableBlock, self).__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels=channels,       #1
            out_channels=channels,      #1
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
    Implements configs.n_blocks of depthwise separable convolutions with kernel size configs.kernel_size and a linear prediction head
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        # load parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.kernel_size = configs.conv_kernel_size
        self.n_blocks = configs.n_blocks

        # model
        self.blocks = nn.Sequential(
            *[DepthwiseSeparableBlock(self.enc_in, self.kernel_size) for _ in range(self.n_blocks)]
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