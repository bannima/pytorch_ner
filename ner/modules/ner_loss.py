#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: loss.py
@time: 2022/2/15 4:40 PM
@desc: 
"""
import torch
import torch.nn as nn

class SeqWiseCrossEntropyLoss():
    ''' implement cross entropy loss for variable seq length in NER task.'''
    def __init__(self):
        self.crossentropy = nn.CrossEntropyLoss()

    def __call__(self,inputs,labels,seq_lengths):
        loss = 0
        # loop each seq for batch data
        for input,label,seq_length in zip(inputs,labels,list(seq_lengths)):
            loss += self.crossentropy(input[:seq_length,],torch.tensor(label[:seq_length],dtype=torch.long))
        return loss

__registered_loss = {
    'SeqWiseCrossEntropyLoss':{
        'cls':SeqWiseCrossEntropyLoss,
        'intro':''
    },
    'BCEWithLogitsLoss':{
        'cls':nn.BCEWithLogitsLoss,
        'intro':''
    },
    'KLDivLoss':{
        'cls':nn.KLDivLoss,
        'intro':''
    },
    'MultiLabelMarginLoss':{
        'cls':nn.MultiLabelMarginLoss,
        'intro':''
    },
    'CrossEntropyLoss':{
        'cls':nn.CrossEntropyLoss,
        'intro':''
    },
    'BCELoss':{
        'cls':nn.BCELoss,
        'intro':''
    }
}

def create_loss(loss_type):
    if loss_type not in __registered_loss:
        raise ValueError("{} not registered, must in {}".format(list(__registered_loss.keys())))
    return __registered_loss[loss_type]['cls']()
