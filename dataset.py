#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_tar, language_src, language_tar, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tar = tokenizer_tar
        self.language_src = language_src
        self.language_tar = language_tar

        # tokenizers.Tokenizer.token_to_id(token: str) -> index in the tokenizer's vocabulary: int
        self.sos_token = torch.tensor([tokenizer_tar.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tar.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tar.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_tar_pair = self.dataset[index]
        src_sentence = src_tar_pair['translation'][self.language_src]
        tar_sentence = src_tar_pair['translation'][self.language_tar]

        # transform the sentences into tokens using tokenizers.Tokenizer.encode() 
        # ... which returns: .ids -> list of token IDs, .tokens -> list of tokens
        encoder_input_tokens = self.tokenizer_src.encode(src_sentence).ids
        decoder_input_tokens = self.tokenizer_tar.encode(tar_sentence).ids

        # add special tokens (sos_token, eos_token and pad_token) to each sentence
        """
            an example with seq_len = 7:

                 position  ->    0    1      2         3         4      5      6
            encoder_input  ->  <s>  The    cat        is  sleeping   </s>  <pad>   (with <s> and </s>)
            decoder_input  ->  <s>  Die  Katze  schlaeft     <pad>  <pad>  <pad>   (only with <s>)
            decoder_label  ->   --  Die  Katze  schlaeft      </s>  <pad>  <pad>   (only with </s>)
        """
        


    