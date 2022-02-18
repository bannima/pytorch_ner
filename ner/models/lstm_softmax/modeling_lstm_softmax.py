#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: modeling_lstm_softmax.py
@time: 2022/2/11 11:09 AM
@desc: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

class LSTMSoftamx(nn.Module):
    def __init__(self,vocab_size,output_size,word_vectors=None,fine_tuned=True,hidden_size=100,num_lstm_layers=1,bidirectional=True,predict_logits=True,dropout=0.1):
        '''
        :param vocab_size: vocab size for given dataset,using for embedding
        :param output_size: output size for ner label
        :param word_vectors: initialize word vector weights, size = (vocab_size,embedding_size),where embedding_size ==hidden_size
        :param fine_tuned: fine tune the word vector weights or not
        :param hidden_size: embedding size of word vectors, as well as the lstm input size
        :param num_lstm_layers: number of lstm layer
        :param bidirectional: is bidirectional lstm or not
        '''
        super(LSTMSoftamx, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_lstm_layers = num_lstm_layers
        self.output_size = output_size
        self.predict_logits = predict_logits
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(vocab_size,hidden_size)
        if word_vectors is not None:
            self.embedding.weight = nn.Parameter(word_vectors,requires_grad=fine_tuned)

        self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_lstm_layers,bidirectional=self.bidirectional)

        #if bilstm, double the input feature size of fully connected layer
        self.fc = nn.Linear(hidden_size*(int(bidirectional)+1),output_size)

    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.num_lstm_layers*(int(self.bidirectional)+1),
                             batch_size,self.hidden_size)
        return hidden

    def forward(self,seq_tensor,seq_lengths):
        #seq_tensor,seq_lengths = input # seq_tensor = B x S
        seq_tensor = seq_tensor.t() # seq_tensor = S x B
        batch_size = seq_tensor.size(1)

        hidden = self._init_hidden(batch_size)
        cell = self._init_hidden(batch_size)
        try:
            embedd = self.embedding(seq_tensor) #  S x B x E
        except:
            print("rase")
        # pack them up nicely
        lstm_input = pack_padded_sequence(
            embedd, seq_lengths.data.cpu().numpy()
        )

        # to compact weights again call flatten paramters
        output, (final_hidden_state, final_cell_state) = self.lstm(lstm_input, (hidden, cell))
        output,length = pad_packed_sequence(output) # output = S x B x E
        logits = self.dropout(output.view(-1,self.hidden_size*(int(self.bidirectional)+1)))
        logits = self.fc(logits) #double hidden size when bilstm
        logits = logits.view(-1,batch_size,self.output_size).transpose(0,1) # S x B x L -> B x S x L, L means output label size

        if self.predict_logits:
            return logits  # B x S x L
        return torch.argmax(logits,dim=-1) # final predicts, B x S x L -> B x S, B=32,S=50,L=35
