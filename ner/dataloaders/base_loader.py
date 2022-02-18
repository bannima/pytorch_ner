#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: base_loader.py
@time: 2022/2/15 11:22 AM
@desc: 
"""
from abc import ABCMeta,abstractmethod

class BaseLoader(metaclass=ABCMeta):
    def __init__(self,batch_size):
        self.batch_size = batch_size

    @property
    @abstractmethod
    def train_loader(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def valid_loader(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def test_loader(self):
        raise NotImplementedError()

    def raw_to_vector(self, inputs, **kwargs):
        ''' convert the raw input and label to numerical vector'''
        return inputs

    def vector_to_raw(self, inputs, **kwargs):
        ''' convert the numerical vector to raw input and label '''
        return inputs

    @property
    def info(self):
        ''' return dataset information where model init needs '''
        return None