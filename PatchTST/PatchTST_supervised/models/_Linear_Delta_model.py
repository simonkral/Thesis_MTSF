"""
Elements from:

Github:                 https://github.com/hanlu-nju/channel_independent_MTSF,
specifically the file:  https://github.com/hanlu-nju/channel_independent_MTSF/blob/main/models/Linear.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Linear delta model:

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in # Number of channels

        # Delta hyperparam. 

        self.Linear_CI = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_CD = nn.Linear(self.seq_len * self.enc_in, self.pred_len * self.enc_in)
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]      

        out_CI = self.Linear_CI(x.permute(0, 2, 1)).permute(0, 2, 1)

        bz = x.size(0)
        out_CD = self.Linear_CD(x.view(bz, -1)).reshape(bz, self.pred_len, self.enc_in)

        out_hybrid = out_CD + out_CI

        return out_hybrid # [Batch, Output length, Channel]