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
from ner.modules.analyzer import SingleTaskExpAnalyzer

def cluener_lstm_softmax_exp(hypers):
    label_type = 'bioes' # label type to preprocess cluener dataset, must in [bio,bioes]
    result_path = os.path.join(os.path.dirname(__file__),'results')

    # 1.load cluener dataset
    cluener = create_dataloader('CLUENER',batch_size=hypers.batch_size,label_type=label_type,result_path=result_path)

    # 2.initialize the lstm_softmax model
    lstm_softmax = create_ner_model(ner_model='LSTMSoftmax',vocab_size=cluener.vocab_size,\
                        output_size=cluener.label_size,word_vectors=None,hidden_size=100,bidirectional=True)

    # 3.prepare the trainer
    trainer = Trainer(
        model = lstm_softmax,
        dataloaders = (cluener.train_loader,cluener.valid_loader,cluener.test_loader),
        raw_to_vector=cluener.raw_to_vector,
        vector_to_raw=cluener.vector_to_raw,
        result_path = result_path,
        hypers = hypers
    )
    trainer.fit()

    # 4.analysis experiment results
    cur_dir = os.path.dirname(__file__)
    analyzer = SingleTaskExpAnalyzer(os.path.join(cur_dir,trainer.epoch_stats_file))
    analyzer.analysis_experiment(exp_result_dir=trainer.exp_result_dir,title="Cluener_LstmSoftmax_Experiment")

if __name__ == '__main__':
    hypers = parse_parmas()
    cluener_lstm_softmax_exp(hypers)




