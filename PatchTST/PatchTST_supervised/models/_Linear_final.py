"""
Elements from:

Github:                 https://github.com/cure-lab/LTSF-Linear,
specifically the file:  https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py
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
        self.channel_handling = configs.channel_handling

        if self.channel_handling == "CI_loc":
            self.Linear_CI = nn.ModuleList()
            for i in range(self.enc_in):
                self.Linear_CI.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_CI = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_CD = nn.Linear(self.seq_len * self.enc_in, self.pred_len * self.enc_in)
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]  

        # CI component
        if self.channel_handling == "CI_loc":
            out_CI = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.enc_in):
                out_CI[:,:,i] = self.Linear_CI[i](x[:,:,i])
        else:
            out_CI = self.Linear_CI(x.permute(0,2,1)).permute(0,2,1)    

        # CD component
        bz = x.size(0)
        out_CD = self.Linear_CD(x.view(bz, -1)).reshape(bz, self.pred_len, self.enc_in)

        # final output
        if self.channel_handling == "CI_glob" or self.channel_handling == "CI_loc":
            out = out_CI
        elif self.channel_handling == "CD":
            out = out_CD
        elif self.channel_handling == "Delta":
            out = out_CI + out_CD
        
        return out # [Batch, Output length, Channel]