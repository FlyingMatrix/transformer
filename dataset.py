#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_tar, language_src, language_tar, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tar = tokenizer_tar
        self.language_src = language_src
        self.language_tar = language_tar
        self.seq_len = seq_len

        # tokenizers.Tokenizer.token_to_id(token: str) -> index in the tokenizer's vocabulary: int
        # [SOS], [EOS], [PAD] are built into the vocabulary via WordLevelTrainer in train.py
        # self.sos_token, self.eos_token, self.pad_token -> torch.tensor
        self.sos_token = torch.tensor([tokenizer_tar.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tar.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tar.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_tar_pair = self.dataset[index] 
        # src_tar_pair -> {'id': string, 'translation': dict}, translation -> {"en": "I love you", "it": "Ti amo"}
        src_sentence = src_tar_pair['translation'][self.language_src]
        tar_sentence = src_tar_pair['translation'][self.language_tar]

        # transform the sentences into tokens using tokenizers.Tokenizer.encode() 
        # which returns: .ids -> list of token IDs, .tokens -> list of tokens
        encoder_input_tokens = self.tokenizer_src.encode(src_sentence).ids # token IDs: List
        decoder_input_tokens = self.tokenizer_tar.encode(tar_sentence).ids # token IDs: List

        # add special tokens (sos_token, eos_token and pad_token) to each sentence
        """
            an example with seq_len = 7:

                 position  ->    0    1      2         3         4      5      6
            encoder_input  ->  <s>  The    cat        is  sleeping   </s>  <pad>   (with <s> and </s>)
            decoder_input  ->  <s>  Die  Katze  schlaeft     <pad>  <pad>  <pad>   (only with <s>)
            decoder_label  ->   --  Die  Katze  schlaeft      </s>  <pad>  <pad>   (only with </s>)

            the decoder_input normally has one offset to the right for training

            for example:
                the decoder sees <s> and should learn to predict "Die"
                then the decoder sees "Die" and should predict "Katze" and so on
                at the end, the model is trained to predict </s>
        """
        num_paddings_encoder = self.seq_len - len(encoder_input_tokens) - 2
        num_paddings_decoder = self.seq_len - len(decoder_input_tokens) - 1

        # make sure the number of paddings is not negative, otherwise the sentence is too long
        if num_paddings_encoder < 0 or num_paddings_decoder < 0:
            raise ValueError("sentence is too long!")
        
        # add sos_token and eos_token to encoder_input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_paddings_encoder, dtype=torch.int64)
            ],
            dim=0
        )

        # add sos_token to the decoder_input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * num_paddings_decoder, dtype=torch.int64)
            ],
            dim=0
        )

        # add eos_token to the decoder_label
        decoder_label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_paddings_decoder, dtype=torch.int64)
            ],
            dim=0
        )

        # double check the size of the tensors to guarantee they are all seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert decoder_label.size(0) == self.seq_len

        # encoder_mask: to prevent attention to padding tokens during multi-head self-attention
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # True, False -> 1, 0
        # encoder_mask -> (1, 1, seq_len)

        # decoder_mask: to prevent the model from looking ahead (future tokens must not be attended to)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)) # (1, seq_len) & (1, seq_len, seq_len)

        # decoder_mask -> (1, seq_len, seq_len)
        """
            decoder_mask -> (1, seq_len, seq_len)
            which combines causality (no future attending) + padding (ignore pad tokens)

            The value at [0, i, j] = decoder_mask[0, j] & causal_mask[0, i, j]

            Bitwise & behaves as below:
            1 & True -> True
            0 & True -> False
            1 & False -> False
            0 & False -> False

            Hence: value = True if:
                       decoder_input[j] != pad_token (i.e. real token) AND it's a valid causal position (j <= i)
                   otherwise value = False
        """

        return {
            "encoder_input": encoder_input, # torch.tensor: (seq_len)
            "decoder_input": decoder_input, # torch.tensor: (seq_len)
            "decoder_label": decoder_label, # torch.tensor: (seq_len)
            "src_sentence": src_sentence,
            "tar_sentence": tar_sentence,
            "encoder_mask": encoder_mask, # torch.tensor: (1, 1, seq_len)
            "decoder_mask": decoder_mask  # torch.tensor: (1, seq_len, seq_len)
        }


def causal_mask(size): # generate a causal attention mask
    """
        causal attention mask is to prevent the model from looking ahead (future tokens must not be attended to)
    """
    causal_mask = torch.triu(torch.ones((1, size, size), dtype=torch.int), diagonal=1)
    return causal_mask == 0


