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
from ner.models.bert_softmax.modeling_bert_softmax import BERTSoftmax
from ner.models.bert_crf.modeling_bert_crf import BERTCRF

__registerd_models = {
    'LSTMSoftmax':{
        'cls':LSTMSoftamx,
        'intro':'LSTM+Softmax'
    },
    'LSTMCRF': {
        'cls': LSTMCRF,
        'intro': "LSTM+CRF",
    },
    'BERTSoftmax':{
        'cls':BERTSoftmax,
        'intro':"Bert+Softmax"
    },
    'BERTCRF':{
        'cls':BERTCRF,
        'intro':"Bert+CRF"
    }
}

def create_ner_model(ner_model,*args,**kwargs):
    if ner_model not in __registerd_models:
        raise ValueError("{} not in registered models, must in {}".format(ner_model,list(__registerd_models.keys())))
    return __registerd_models[ner_model]['cls'](*args,**kwargs)