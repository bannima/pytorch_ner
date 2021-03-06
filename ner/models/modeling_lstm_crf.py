#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: modeling_lstm_crf.py
@time: 2022/2/9 7:56 PM
@desc: 
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torchcrf import CRF

class LSTMCRF(nn.Module):
    ''' Implementation of LSTM + CRF '''

    def __init__(self,vocab_size,output_size,word_vectors=None,fine_tuned=True,hidden_size=100,num_lstm_layers=1,bidirectional=True,predict_logits=True,dropout=0.1,output_loss=True):
        super(LSTMCRF, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_lstm_layers = num_lstm_layers
        self.output_size = output_size
        self.predict_logits = predict_logits
        self.dropout = nn.Dropout(p=dropout)

        # for lstm_crf, model output loss
        self.output_loss = output_loss

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if word_vectors is not None:
            self.embedding.weight = nn.Parameter(word_vectors, requires_grad=fine_tuned)

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_lstm_layers,
                            bidirectional=self.bidirectional)

        # if bilstm, double the input feature size of fully connected layer
        self.fc = nn.Linear(hidden_size * (int(bidirectional) + 1), output_size)

        self.crf = CRF(num_tags=self.output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_lstm_layers * (int(self.bidirectional) + 1),
                             batch_size, self.hidden_size)
        return hidden

    def forward(self,inputs, labels):
        seq_tensor,mask = inputs # seq_tensor = B x S
        seq_tensor = seq_tensor.t()  # seq_tensor = S x B
        batch_size = seq_tensor.size(1)

        # get seq_lengths from mask
        seq_lengths = torch.sum(mask, dim=1)
        seq_lengths = seq_lengths.t()

        hidden = self._init_hidden(batch_size)
        cell = self._init_hidden(batch_size)
        embedd = self.embedding(seq_tensor)  # S x B x E

        # pack them up nicely
        lstm_input = pack_padded_sequence(
            embedd, seq_lengths.data.cpu().numpy()
        )
        # to compact weights again call flatten paramters
        output, (final_hidden_state, final_cell_state) = self.lstm(lstm_input, (hidden, cell))
        output, length = pad_packed_sequence(output)  # output = S x B x E

        logits = self.fc(self.dropout(output)) # double hidden size when bilstm
        logits = logits.transpose(0,1) # S x B x L -> B x S x L, L means output label size

        loss = None
        logits = logits.transpose(0, 1)  # crf emission requrires (seq_length, batch_size, num_tags) format
        mask = mask.transpose(0, 1)  # mask: (seq_length, batch_size)
        # no inference should calc loss
        if labels is not None:
            labels = labels.transpose(0, 1)  # labels: BxS -> SxB
            log_likelihood = self.crf(logits,labels,mask) #crf output the log likelihood
            loss = -1 * log_likelihood

        predictions = self.crf.decode(logits,mask) # batch_size x seq_size
        return predictions,loss










