import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Implements a depthwise separable convolution with kernel size 5 with a linear head
    """

    def __init__(self, configs, kernel_size=5):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels=self.enc_in,
            out_channels=self.enc_in,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.enc_in
        )
        
        # 1x1 (pointwise convolution) convolution to mix the channels
        self.pointwise = nn.Conv1d(
            in_channels=self.enc_in,
            out_channels=self.enc_in,
            kernel_size=1
        )
        
        # linear head
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Sequence Length, Channels] -> [Batch, Channels, Sequence Length]
        x = x.permute(0, 2, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.Linear(x)
        # back to [Batch, Sequence Length, Channels]
        x = x.permute(0, 2, 1)
        return x # [Batch, Output length, Channel]