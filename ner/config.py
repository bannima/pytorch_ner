#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: config.py
@time: 2022/2/11 11:16 AM
@desc: 
"""
import os
import logzero
from ner.modules.utils import which_day
from logzero import logger

#project path
project_path = os.path.dirname(os.path.dirname(__file__))

#data path
dataset_path = os.path.join(project_path, 'dataset')

#logger
log_path = os.path.join(project_path, 'log')
log_file = os.path.join(log_path,"all_{}.log".format(which_day()))
logzero.logfile(log_file,maxBytes=1e6,backupCount=3)

#random seed
random_seed = 9527