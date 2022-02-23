#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: modeling_bert_softmax.py
@time: 2022/2/22 9:36 AM
@desc: 
"""

import torch
import torch.nn as nn
from ner.pretrain_models import create_pretrained_model,create_pretrained_tokenizer

class BERTSoftmax(nn.Module):
    '''BERT + softmax layer'''
    def __init__(self,
                 max_length,
                 output_size,
                 pretrain_model_type,
                 dropout=0.2,
                 output_loss=False
                 ):
        '''
        :param max_length:
        :param output_size:
        :param pretrain_model_type:
        :param predict_logits:
        :param dropout:
        '''
        super(BERTSoftmax, self).__init__()
        self.output_loss = False
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
        self.bert = create_pretrained_model(model_type=pretrain_model_type)
        self.fc = nn.Linear(768,output_size)

    def forward(self,inputs,labels):
        bert_output = self.bert(*inputs)[0] # batch_size x seq_len x hidden_size(768)

        logits = self.fc(bert_output) # batch_size x seq_len x output_size

        return logits



