#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: metrics.py
@time: 2022/2/15 4:46 PM
@desc: 
"""
from functools import partial
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score,auc,roc_auc_score
from sklearn.metrics import ndcg_score

__registered_metrics = {
    'classification':{
        'f1':f1_score,
        'precision':precision_score,
        'accuracy':accuracy_score,
        'roc_auc_score':roc_auc_score
    }
}

def create_metrics(metric_type):
    if metric_type not in __registered_metrics:
        raise ValueError("{} not registered, must in {}".format(list(__registered_metrics.keys())))
    return __registered_metrics[metric_type]
