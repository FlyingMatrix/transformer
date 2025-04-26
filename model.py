#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    A pytorch implementation of transformer model
"""

import torch
import torch.nn as nn
import math
import numpy as np


class InputEmbedding(nn.Module):

    def __init__(self, dim_model, vocab_size):
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim_model)

    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, dim_model)
        return self.embedding(x) * math.sqrt(self.dim_model) 
    
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, dim_model, seq_len, dropout): # here the seq_len is the maximum length of the sentence
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # the output of positional encoding is a matrix of shape (seq_len, dim_model)
        pe = torch.ones(seq_len, dim_model)
        for pos in range(seq_len): 
            for i in range(dim_model):
                pe[pos][i] = (
                    np.sin(pos / (10000 ** (i / dim_model))) if i % 2 == 0 else np.cos(pos / (10000 ** ((i - 1) / dim_model)))
                )
        # add a batch dimension
        pe = pe.unsqueeze(0) # pe -> (1, seq_len, dim_model)
        # register as a buffer
        self.register_buffer('positional_encoding', pe)

    def forward(self, x): # x -> (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)
        

