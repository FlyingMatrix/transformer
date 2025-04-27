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

    def forward(self, x): # x -> (batch_size, seq_len, dim_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)
        

class LayerNormalization(nn.Module): # can also use layer_norm = nn.LayerNorm(dim_model)

    def __init__(self, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        layernorm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return layernorm
    

class FeedForwardBlock(nn.Module):
    
    def __init__(self, dim_model, dim_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_ff, dim_model) # W2 and B2

    def forward(self, x):
        # (batch_size, seq_len, dim_model) -> (batch_size, seq_len, dim_ff) -> (batch_size, seq_len, dim_model)
        feedforward = self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # torch.relu(x) = max(0, x)
        return feedforward
    

