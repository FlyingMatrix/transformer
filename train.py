#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Hugging Face tokenizers
"""
    Hugging Face tokenizers best for: Transformer-based models (BERT, GPT, etc.)
"""
import torchtext.datasets as datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


# build special token list
unknown_token="[UNK]"
padding_token = "[PAD]"
start_of_sentence = "[SOS]"
end_of_sentence = "[EOS]"
special_tokens = [unknown_token, padding_token, start_of_sentence, end_of_sentence]


def get_all_sentences():
    
    pass


def get_tokenizer(config, dataset, language): 

    """
        What the Tokenizer does:
        > It tokenizes text (splits it into tokens).
        > It maps tokens to IDs based on a vocabulary.
        > If a token isn't in the vocabulary, it uses the unk_token.
    """
    
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    
    if not Path.exists(tokenizer_path):
        # create a new tokenizer using the WordLevel model from Hugging Face tokenizers library
        tokenizer = Tokenizer(WordLevel(unk_token=unknown_token)) # specifies the token as "[UNK]" when not found in the vocabulary
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=special_tokens, min_frequency=2)
        # build the vocabulary
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

