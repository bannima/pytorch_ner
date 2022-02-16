#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: __init__.py.py
@time: 2022/2/11 11:11 AM
@desc: 
"""
from ner.dataloaders.cluener_loader import CLUENERDataloader

__registered_dataloaders = {
    "CLUENER":{
        'cls':CLUENERDataloader,
        'intro':""
    }
}

def create_dataloader(dataset,batch_size,*args,**kwargs):
    if dataset not in __registered_dataloaders:
        raise ValueError("#ERROR not recognized dataset {}, dataset must in {}".format(dataset,list(__registered_dataloaders.keys())))

    data_loader = __registered_dataloaders[dataset]['cls'](batch_size,*args,**kwargs)
    return data_loader
