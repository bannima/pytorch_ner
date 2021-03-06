#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: cluener_loader.py
@time: 2022/2/11 11:11 AM
@desc: 
"""
import os
from collections import Counter
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.utils import Bunch
from sklearn.preprocessing import LabelBinarizer
from gensim.models.word2vec import KeyedVectors
from torch.utils.data import Dataset,DataLoader
from ner.config import logger, dataset_path,random_seed
from ner.modules.utils import read_from_json
from ner.dataloaders.base_loader import BaseLoader
from ner.modules.utils import get_label_binarizer,load_word_vectors
from ner.modules.utils import convert_to_bert_batch_data

def transform_cluerner_label_with_bio(record):
    r""" transform cluener dataset record with b,i,o label """

    text = record['text']
    label = record['label']
    if pd.isna(label): #test set has no label
        return None
    bio_label = ['O'] * len(text)
    for type in label:
        for entity in label[type]:
            for (start_pos, end_pos) in label[type][entity]:
                bio_label[start_pos:end_pos + 1] = ['B-{}'.format(type)] + ['I-{}'.format(type)] * (end_pos - start_pos)
    return '|'.join(bio_label)

def transform_cluerner_label_with_bioes(record):
    r""" transform cluener dataset record with b,i,o,e,s label """
    text = record['text']
    label = record['label']
    if pd.isna(label):  # test set has no label
        return None
    bio_label = ['O'] * len(text)
    for type in label:
        for entity in label[type]:
            for (start_pos, end_pos) in label[type][entity]:
                if (1 + end_pos - start_pos) == 1:
                    bio_label[start_pos:end_pos + 1] = ['S-{}'.format(type)]
                elif (1 + end_pos - start_pos) == 2:
                    bio_label[start_pos:end_pos + 1] = ['B-{}'.format(type)] + ['E-{}'.format(type)]
                elif (1 + end_pos - start_pos) >= 3:
                    bio_label[start_pos:end_pos + 1] = ['B-{}'.format(type)] + ['I-{}'.format(type)] * (
                                1 + end_pos - start_pos - 2) + ['E-{}'.format(type)]
    return "|".join(bio_label)

def build_vocab(dataset):
    max_length = 0
    vocab_counter = Counter()
    for text in dataset:
        max_length = max(max_length,len(list(text)))
        vocab_counter += Counter(list(text))
    word2idx = {word:idx+1 for idx,word in enumerate(vocab_counter.keys())} #note that idx start from 1, 0 should be defined as OOV(out of vocabulary)
    idx2word = {idx:word for word,idx in word2idx.items()}
    return len(vocab_counter)+1,word2idx,idx2word,max_length #note that the vocab size should add 1,because the word2idx starts from 1

class CLUENERDataset(Dataset):
    def __init__(self,dataset):
        super(CLUENERDataset, self).__init__()
        self.dataset = dataset.to_dict(orient='record')

    def __getitem__(self, idx):
        # x = Bunch()
        # x.text = self.dataset[idx]['text']
        # x.label = self.dataset[idx]['label']
        if self.dataset[idx]['label'] is None:
            return self.dataset[idx]['text']
        return self.dataset[idx]['text'],self.dataset[idx]['label']

    def __len__(self):
        return len(self.dataset)

class CLUENERDataloader(BaseLoader):
    def __init__(self,batch_size,label_type='bio',*args,**kwargs):
        super(CLUENERDataloader, self).__init__(batch_size=batch_size)
        assert label_type in ['bio','bioes']
        self._train_loader,self._valid_loader,self._test_loader = None,None,None
        self.result_path = kwargs.get('result_path')
        self.preprocess(label_type)

    def preprocess(self,label_type):
        """construct the cluener dataset"""
        # 1. load cluener dataset
        cluener_train_file = os.path.join(dataset_path,'zh/clue/train.json')
        cluener_train = pd.DataFrame(read_from_json(cluener_train_file))
        cluener_test_file = os.path.join(dataset_path, 'zh/clue/test.json')
        cluener_test = pd.DataFrame(read_from_json(cluener_test_file))

        # preprocess
        cluener_train,cluener_val = train_test_split(cluener_train,test_size=0.05,random_state=random_seed)
        cluener_train['type'] = 'train'
        cluener_val['type'] = 'val'
        cluener_test['type'] = 'test'

        self.cluener = pd.concat([cluener_train,cluener_val,cluener_test],axis=0)

        #transform the cluener label to labeltype(bio or bioes)
        if label_type =='bio':
            self.cluener['label'] = self.cluener.apply(transform_cluerner_label_with_bio,axis=1)
        elif label_type=='bioes':
            self.cluener['label'] = self.cluener.apply(transform_cluerner_label_with_bioes,axis=1)

        logger.info("# CLUENER dataset label transformed. ")

        # 2. numerical the original label and embedding the text
        # fit the multi label binarizer
        label_binarizer_path = os.path.join(self.result_path,'cluener_{}.mlb'.format(label_type))
        self.label_binarizer = get_label_binarizer(label_binarizer_path,self.cluener.loc[self.cluener['type']!='test']['label'].apply(lambda label:label.split('|')))
        self.label_size  = len(self.label_binarizer.classes_)
        self.label2idx = {label:idx for idx,label in enumerate(self.label_binarizer.classes_)}
        self.idx2label = {idx:label for label,idx in self.label2idx.items()}
        logger.info("# CLUENER label binarizer loaded. ")

        # build the vocabulary
        self.vocab_size,self.word2idx,self.idx2word,self.max_length = build_vocab(self.cluener['text'])
        logger.info("# CLUENER dataset preprocessed. ")

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(
                dataset=CLUENERDataset(self.cluener[self.cluener['type']=='train']),
                num_workers=1,
                batch_size=self.batch_size,
                shuffle=True
            )
        return self._train_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(
                dataset = CLUENERDataset(self.cluener[self.cluener['type']=='test']),
                num_workers=1,
                batch_size=self.batch_size,
                shuffle=True
            )
        return self._test_loader

    @property
    def valid_loader(self):
        if self._valid_loader is None:
            self._valid_loader = DataLoader(
                dataset = CLUENERDataset(self.cluener[self.cluener['type']=='val']),
                batch_size=self.batch_size,
                num_workers=1,
                shuffle=True
            )
        return self._valid_loader

    def raw_to_vector(self, inputs, labels, tokenizer=None, max_length=None):
        """ convert the raw text into idx list and transform to torch tensor.
            note that the variable length of text sequence should be filled with zero
            and sort by the sequence length.

            vectorize the raw label and transform to torch tensor,
            permute the label sequence order by the text perm order.
        """

        #calc variable seq lengths
        seq_lengths = torch.LongTensor([len(text) for text in inputs])

        # inference has not labels
        if labels is not None:
            # vectorize the label
            labels = [[self.label2idx[l] for l in label.split('|')] for label in labels]
            
            # fill vairable length labels with -1, note that -1 are meaningless in cluener label
            label_tensor = torch.full((len(inputs), seq_lengths.max()), -1)
            for idx, (seq_label, seq_len) in enumerate(zip(labels, seq_lengths)):
                label_tensor[idx, :seq_len] = torch.LongTensor(seq_label)

        if not tokenizer:
            # vectorize the char sequence
            inputs = [[self.word2idx[char] for char in list(x)] for x in inputs]

            #fill vairable length input tensors with zero, note that inputs are encoded from 1, 0 means OOV(out-of-vocabulary)
            seq_tensor = torch.zeros((len(inputs), seq_lengths.max())).long()
            for idx, (seq_input, seq_len) in enumerate(zip(inputs, seq_lengths)):
                seq_tensor[idx, :seq_len] = torch.LongTensor(seq_input)

            # sort tensors by their length
            sorted_seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            seq_tensor = seq_tensor[perm_idx]

            #calc attention mask, note that seq tensor are from 1, 0 means OOV
            mask = torch.where(seq_tensor>0,1,0).type(torch.uint8)

            # inference has not labels
            if labels is not None:
                # also sort label in the same order
                label_tensor = label_tensor[perm_idx]
                return (seq_tensor,mask),label_tensor
            return (seq_tensor,mask),None

        #bert based models, should use tokenizer
        else:
            #cluener for token classification should remove [CLS],[SEP] in the preprocess precess
            bert_inputs = convert_to_bert_batch_data(inputs, tokenizer, max_length,add_special_tokens=False)
            return (bert_inputs,label_tensor) if labels else (bert_inputs,None)

    def vector_to_raw(self,labels):
        return ['|'.join([self.idx2label[l] for l in label]) for label in labels]



