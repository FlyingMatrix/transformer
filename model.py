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
        

class LayerNormalization(nn.Module): # can also use layer_norm = nn.LayerNorm(normalized_shape=features)(x)

    def __init__(self, features, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # multiplied
        self.bias = nn.Parameter(torch.zeros(features)) # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        layernorm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return layernorm
    

class FeedForward(nn.Module):
    
    def __init__(self, dim_model, dim_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_ff, dim_model) # W2 and B2

    def forward(self, x):
        # (batch_size, seq_len, dim_model) -> (batch_size, seq_len, dim_ff) -> (batch_size, seq_len, dim_model)
        feedforward = self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # torch.relu(x) = max(0, x)
        return feedforward
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self, dim_model, num_head, dropout):
        super().__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        # make sure that dim_model is divisible by num_head
        assert dim_model % num_head == 0, "dim_model is not divisible by num_head"

        self.d_k = dim_model // num_head # dimension of vector for each head
        self.w_q = nn.Linear(dim_model, dim_model, bias=False) # Wq
        self.w_k = nn.Linear(dim_model, dim_model, bias=False) # Wk
        self.w_v = nn.Linear(dim_model, dim_model, bias=False) # Wv
        self.w_o = nn.Linear(dim_model, dim_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        # query, key, value -> (batch_size, num_head, seq_len, d_k)
        d_k = query.shape[-1]
        # calculate attention_scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # attention_scores -> (batch_size, num_head, seq_len, seq_len)
        # apply mask
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        # softmax
        attention_scores = attention_scores.softmax(dim=-1)
        # dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # calculate attention
        attention = attention_scores @ value # attention -> (batch_size, num_head, seq_len, d_k)
        return attention, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_len, dim_model) -> (batch_size, seq_len, dim_model)
        key = self.w_k(k)   # (batch_size, seq_len, dim_model) -> (batch_size, seq_len, dim_model)
        value = self.w_v(v) # (batch_size, seq_len, dim_model) -> (batch_size, seq_len, dim_model)

        # (batch_size, seq_len, dim_model) -> (batch_size, seq_len, num_head, d_k) -> (batch_size, num_head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1, 2)

        # calculate attention
        attention, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # attention -> (batch_size, num_head, seq_len, d_k)
        # attention_scores -> (batch_size, num_head, seq_len, seq_len)

        # combine all the heads together
        attention = attention.transpose(1, 2).contiguous().view(attention.shape[0], -1, self.num_head * self.d_k)
        # (batch_size, num_head, seq_len, d_k) -> (batch_size, seq_len, dim_model)

        # multiply by Wo
        attention = self.w_o(attention) # attention -> (batch_size, seq_len, dim_model)

        return attention
    

class ResidualConnection(nn.Module):

    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalized_shape=features)

    def forward(self, x, sublayer): # here sublayer is a function
        residual_connection = x + self.dropout(sublayer(self.norm(x)))
        return residual_connection
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, features: int, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        output = self.residual_connections[1](x, self.feed_forward)
        return output

    
class Encoder(nn.Module):
    
    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers # self.layers is a ModuleList
        self.norm = nn.LayerNorm(normalized_shape=features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        output = self.norm(x)
        return output
    

class DecoderBlock(nn.Module):

    def __init__(self, 
                 self_attention: MultiHeadAttention, 
                 cross_attention: MultiHeadAttention, 
                 feed_forward: FeedForward, 
                 features: int, 
                 dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tar_mask): 
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tar_mask))
        # for cross_attention, query comes from decoder input, while key and value come from encoder output
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        output = self.residual_connections[2](x, self.feed_forward)
        return output


class Decoder(nn.Module):

    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers # self.layers is a ModuleList
        self.norm = nn.LayerNorm(normalized_shape=features)

    def forward(self, x, encoder_output, src_mask, tar_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tar_mask)
        output = self.norm(x)
        return output
        

class ProjectionLayer(nn.Module):

    def __init__(self, dim_model, vocab_size):
        super().__init__()
        self.projection_layer = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        # here x is the output of decoder with a shape of (batch_size, seq_len, dim_model) 
        output = self.projection_layer(x) # output -> (batch_size, seq_len, vocab_size)
        output = torch.softmax(output, dim=-1)
        return output


class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 input_embedding: InputEmbedding,
                 output_embedding: InputEmbedding,
                 src_positional_encoding: PositionalEncoding,
                 tar_positional_encoding: PositionalEncoding,
                 projection_layer: ProjectionLayer
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tar_positional_encoding = tar_positional_encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # src -> (batch_size, seq_len)
        src = self.input_embedding(src) # src -> (batch_size, seq_len, dim_model)
        src = self.src_positional_encoding(src) # src -> (batch_size, seq_len, dim_model)
        encoder_output = self.encoder(src, src_mask) 
        # encoder_output -> (batch_size, seq_len, dim_model)
        return encoder_output

    def decode(self, tar, encoder_output, src_mask, tar_mask):
        # tar -> (batch_size, seq_len)
        tar = self.output_embedding(tar) # tar -> (batch_size, seq_len, dim_model)
        tar = self.tar_positional_encoding(tar) # tar -> (batch_size, seq_len, dim_model)
        decoder_output = self.decoder(tar, encoder_output, src_mask, tar_mask)
        # decoder_output -> (batch_size, seq_len, dim_model)
        return decoder_output
    
    def project(self, decoder_output):
        # decoder_output -> (batch_size, seq_len, dim_model)
        output = self.projection_layer(decoder_output)
        return output # output -> (batch_size, seq_len, vocab_size)


