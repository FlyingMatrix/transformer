#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchtext.datasets as datasets
from datasets import load_dataset

# Hugging Face tokenizers
"""
    Hugging Face tokenizers best for: Transformer-based models (BERT, GPT, etc.)
"""
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


def get_all_sentences(dataset, language):
    
    for item in dataset: 
        # in general, the opus_books dataset is a dictionary structure, so the length of dataset: len(dataset)
        # https://huggingface.co/datasets/Helsinki-NLP/opus_books
        # when a language pair is loaded, each item in the dataset is a dictionary:
        # item -> {'id': string, 'translation': dict}
        # translation -> {"en": "I love you", "it": "Ti amo"}
        yield item['translation'][language] # yield: iterate (one item at a time) over large datasets


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


def get_dataset(config):

    # get dataset
    dataset = load_dataset(path=f"{config['datasource']}",
                           name=f"{config['lang_src']}-{config['lang_tar']}",
                           split='train')

    # build tokenizers
    tokenizer_src = get_tokenizer(config, dataset, config['lang_src'])
    tokenizer_tar = get_tokenizer(config, dataset, config['lang_tar'])

    # split 90% of the dataset for training, 10% for validation
    train_dataset_size = int(0.9 * len(dataset))
    valid_dataset_size = len(dataset) - train_dataset_size
    train_dataset, valid_dataset = random_split(dataset, [train_dataset_size, valid_dataset_size])
        

