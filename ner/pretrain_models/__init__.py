#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: __init__.py.py
@time: 2022/2/22 9:37 AM
@desc: 
"""
import os
import torch
from transformers import BertModel,BertTokenizer
from transformers import RobertaModel,RobertaTokenizer

pretrian_model_path = os.path.dirname(__file__)

__pretrained_models = {
    'bert-base-chinese':{
        'class':BertModel,
        'tokenizer':BertTokenizer,
        'path':'bert-base-chinese/'
    },
    'bert-base-uncased':{
        'class':BertModel,
        'tokenizer':BertTokenizer,
        'path':'bert-base-uncased/'
    },
    'roberta-base':{
        'class':RobertaModel,
        'tokenizer':RobertaTokenizer,
        'path':'roberta-base/'
    }
}

def create_pretrained_model(model_type,retrain_model_path=None):
    ''''load pretrained model'''
    if model_type not in __pretrained_models:
        raise ValueError("Not registered pretrained model type {}, must in {}".format(model_type,list(__pretrained_models.keys())))

    model_path = os.path.join(pretrian_model_path,__pretrained_models[model_type]['path'])
    if retrain_model_path:
        pretrian_model = torch.load(os.path.join(pretrian_model_path,retrain_model_path))
    else:
        pretrian_model = __pretrained_models[model_type]['class'].from_pretrained(model_path)
    return pretrian_model

def create_pretrained_tokenizer(model_type):
    '''load pretrained tokenizer'''
    if model_type not in __pretrained_models:
        raise ValueError("Not registered pretrained tokenizer type {}, must in {}".format(model_type,list(__pretrained_models.keys())))
    model_path = os.path.join(pretrian_model_path,__pretrained_models[model_type]['path'])
    tokenizer = __pretrained_models[model_type]['tokenizer'].from_pretrained(model_path)
    return tokenizer


