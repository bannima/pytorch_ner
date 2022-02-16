#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: __init__.py.py
@time: 2022/2/9 6:07 PM
@desc: 
"""
from ner.models.lstm_crf.modeling_lstm_crf import LSTMCRF
from ner.models.lstm_softmax.modeling_lstm_softmax import LSTMSoftamx

__registerd_models = {
    'LSTMSoftmax':{
        'cls':LSTMSoftamx,
        'intro':'LSTM+Softmax'
    },
    'LSTMCRF': {
        'cls': LSTMCRF,
        'intro': "LSTM+CRF",
    }
}

def create_ner_model(ner_model,*args,**kwargs):
    assert ner_model in __registerd_models
    return __registerd_models[ner_model]['cls'](*args,**kwargs)