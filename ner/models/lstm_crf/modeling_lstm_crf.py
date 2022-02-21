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

    def forward(self,seq_tensor,seq_lengths,labels,is_inference=False):
        # seq_tensor,seq_lengths = input # seq_tensor = B x S
        seq_tensor = seq_tensor.t()  # seq_tensor = S x B
        batch_size = seq_tensor.size(1)

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

        logits = self.dropout(output.view(-1, self.hidden_size * (int(self.bidirectional) + 1)))

        logits = self.fc(logits) # double hidden size when bilstm

        logits = logits.view(-1, batch_size, self.output_size)  # S x B x L, L means output label size


        # # note that labels less than 0 are meaningless label
        # mask = torch.where(labels >= 0, 1, 0).type(torch.uint8)

        # note that seq tensor index less and equal than 0 are meaningless word idx
        mask = torch.where(seq_tensor>0,1,0).type(torch.uint8)

        #train should calc loss
        if not is_inference:
            labels = labels.transpose(0, 1)  # labels: BxS -> SxB

            log_likelihood = self.crf(logits,labels,mask) #crf output the log likelihood
            loss = -1 * log_likelihood
            predictions = self.crf.decode(logits,mask)
            return loss, predictions
        else:
            predictions = self.crf.decode(logits,mask)
            return predictions









