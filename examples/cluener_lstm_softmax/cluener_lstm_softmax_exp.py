#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: cluener_lstm_softmax_exp.py
@time: 2022/2/11 11:16 AM
@desc: 
"""
import os.path

from ner.dataloaders import create_dataloader
from ner.models import create_ner_model
from ner.modules.utils import parse_parmas
from ner.modules.trainer import Trainer
from transformers.models.bert import BertModel,BertTokenizer


def cluener_lstm_softmax_exp(hypers):
    label_type = 'bioes' # label type to preprocess cluener dataset, must in [bio,bioes]
    result_path = os.path.join(os.path.dirname(__file__),'results')

    # 1.load cluener dataset
    cluener_loader = create_dataloader('CLUENER',batch_size=hypers.batch_size,label_type=label_type,result_path=result_path)

    # 2.initialize the lstm_softmax model
    lstm_softmax = create_ner_model(ner_model='LSTMSoftmax',vocab_size=cluener_loader.vocab_size,\
                        output_size=cluener_loader.label_size,word_vectors=None,hidden_size=100,bidirectional=True)

    # 3.prepare the trainer
    trainer = Trainer(
        model = lstm_softmax,
        dataloaders = (cluener_loader.train_loader,cluener_loader.valid_loader,cluener_loader.test_loader),
        data_converter=cluener_loader.data_converter,
        result_path = result_path,
        hypers = hypers
    )
    trainer.fit()

    # 4.analysis experiment results


if __name__ == '__main__':
    hypers = parse_parmas()
    cluener_lstm_softmax_exp(hypers)




