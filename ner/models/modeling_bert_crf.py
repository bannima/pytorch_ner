#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: modeling_bert_crf.py
@time: 2022/2/23 1:40 PM
@desc: 
"""
import torch
import torch.nn as nn
from torchcrf import CRF
from ner.pretrain_models import create_pretrained_model

class BERTCRF(nn.Module):
    '''BERT + CRF layer'''
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
        super(BERTCRF, self).__init__()
        self.output_loss = True # crf should output loss
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
        self.bert = create_pretrained_model(model_type=pretrain_model_type)
        self.fc = nn.Linear(768,output_size)
        self.crf = CRF(num_tags=self.output_size)


    def forward(self,inputs,labels):
        # the 1st place of inputs are mask
        mask = inputs[1].type(torch.uint8)

        bert_output = self.bert(*inputs)[0] # batch_size x seq_len x hidden_size(768)
        logits = self.fc(bert_output) # batch_size x seq_len x output_size

        loss = None
        logits = logits.transpose(0, 1)  # crf emission requrires (seq_length, batch_size, num_tags) format
        mask = mask.transpose(0, 1)  # mask: (seq_length, batch_size)
        # no inference should calc loss
        if labels is not None:
            labels = labels.transpose(0, 1)  # labels: BxS -> SxB
            log_likelihood = self.crf(logits, labels, mask)  # crf output the log likelihood
            loss = -1 * log_likelihood

        predictions = self.crf.decode(logits, mask)  # batch_size x seq_size
        return predictions, loss


