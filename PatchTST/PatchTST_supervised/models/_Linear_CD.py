"""
Elements from:

Github:                 https://github.com/hanlu-nju/channel_independent_MTSF,
specifically the file:  https://github.com/hanlu-nju/channel_independent_MTSF/blob/main/models/Linear.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CD Linear model:

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in # Number of channels
        self.flat_input = True # False = CI, True -> CD

        if self.flat_input:
            self.Linear = nn.Linear(self.seq_len * self.enc_in, self.pred_len * self.enc_in)
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]        
        if self.flat_input:
            bz = x.size(0)
            x = self.Linear(x.view(bz, -1)).reshape(bz, self.pred_len, self.enc_in)
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x # [Batch, Output length, Channel]