#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: cluener_bert_softmax_exp.py
@time: 2022/2/22 10:05 AM
@desc: 
"""
import os.path
from functools import partial
from ner.dataloaders import create_dataloader
from ner.models import create_ner_model
from ner.modules.utils import parse_parmas
from ner.modules.trainer import Trainer
from ner.modules.analyzer import SingleTaskExpAnalyzer
from ner.pretrain_models import create_pretrained_tokenizer


def cluener_bert_softmax_exp(hypers):
    # label type to preprocess cluener dataset, must in [bio,bioes]
    label_type = 'bioes'
    # chinese cluener dataset should use bert-base-chinese
    pretrain_model_type = 'bert-base-chinese'

    result_path = os.path.join(os.path.dirname(__file__),'results')

    # 1.load cluener dataset
    cluener = create_dataloader('CLUENER',batch_size=hypers.batch_size,label_type=label_type,result_path=result_path)

    #max seq length
    max_length = cluener.max_length

    # 2.initialize the bert_softmax model, tokenizer
    bert_tokenizer = create_pretrained_tokenizer(model_type=pretrain_model_type)

    bert_softmax = create_ner_model(ner_model='BERTSoftmax',
                                    max_length=max_length,
                                    output_size=cluener.label_size,
                                    pretrain_model_type=pretrain_model_type,
                                    dropout=0.2)

    # 3.prepare the trainer
    trainer = Trainer(
        model = bert_softmax,
        dataloaders = (cluener.train_loader,cluener.valid_loader,cluener.test_loader),
        raw_to_vector=partial(cluener.raw_to_vector,tokenizer=bert_tokenizer,max_length=max_length),
        vector_to_raw=cluener.vector_to_raw,
        result_path = result_path,
        hypers = hypers
    )
    trainer.fit()

    # 4.analysis experiment results
    cur_dir = os.path.dirname(__file__)
    analyzer = SingleTaskExpAnalyzer(os.path.join(cur_dir,trainer.epoch_stats_file))
    analyzer.analysis_experiment(exp_result_dir=trainer.exp_result_dir,title="Cluener_BERTSoftmax_Experiment")

if __name__ == '__main__':
    hypers = parse_parmas()
    cluener_bert_softmax_exp(hypers)




