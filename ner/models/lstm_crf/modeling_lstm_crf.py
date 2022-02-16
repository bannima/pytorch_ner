#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: modeling_lstm_crf.py
@time: 2022/2/9 7:56 PM
@desc: 
"""

import torch
import torch.nn as nn
from torchcrf import CRF

class LSTMCRF(nn.Module):
    ''' Implementation of LSTM + CRF '''

    def __init__(self,input_size,hidden_size=200,num_layers=10,bidirectional=True,output_size=10):
        super(LSTMCRF, self).__init__()

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional)
        self.crf = CRF(num_tags=output_size,batch_first=False)

    def forward(self,x):
        x = self.lstm(x)
        x = self.crf(x)
        return x








