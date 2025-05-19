#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Hugging Face tokenizers
"""
    Hugging Face tokenizers best for: Transformer-based models (BERT, GPT, etc.)
"""
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


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


def get_tokenizer(config, dataset, language): # convert sentences in dataset to sequences of token IDs

    """
        What the Tokenizer does:
        > It tokenizes text (splits it into tokens).
        > It maps tokens to IDs based on a vocabulary.
        > If a token isn't in the vocabulary, it uses the unk_token.
    """
    
    tokenizer_path = Path(config['tokenizer_file'].format(language)) # Path(): convert string into a path object
    
    if not tokenizer_path.exists():
        # create a new tokenizer using the WordLevel model from Hugging Face tokenizers library
        tokenizer = Tokenizer(WordLevel(unk_token=unknown_token)) # initialize WordLevel tokenizer
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=special_tokens, min_frequency=2)
        # build the vocabulary
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) # tokenizer_{0}.json in config
    return tokenizer


def get_dataset(config):

    # get dataset
    dataset_raw = load_dataset(path=f"{config['datasource']}",
                               name=f"{config['lang_src']}-{config['lang_tar']}", # language pair
                               split='train')

    # build tokenizers
    tokenizer_src = get_tokenizer(config, dataset_raw, config['lang_src']) # token IDs: List
    tokenizer_tar = get_tokenizer(config, dataset_raw, config['lang_tar']) # token IDs: List

    # split 90% of the dataset for training, 10% for validation
    train_dataset_size = int(0.9 * len(dataset_raw))
    valid_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, valid_dataset_raw = random_split(dataset_raw, [train_dataset_size, valid_dataset_size])
        
    train_dataset = BilingualDataset(train_dataset_raw, 
                                     tokenizer_src, 
                                     tokenizer_tar, 
                                     config['lang_src'], 
                                     config['lang_tar'],
                                     config['seq_len'])
    
    valid_dataset = BilingualDataset(valid_dataset_raw, 
                                     tokenizer_src, 
                                     tokenizer_tar, 
                                     config['lang_src'], 
                                     config['lang_tar'],
                                     config['seq_len'])
    
    # find the maximum length of each sentence in source and target language
    max_len_src = 0
    max_len_tar = 0

    for item in dataset_raw:
        sentence_src = item['translation'][config['lang_src']]
        sentence_tar = item['translation'][config['lang_tar']]

        ids_src = tokenizer_src.encode(sentence_src).ids
        ids_tar = tokenizer_tar.encode(sentence_tar).ids

        max_len_src = max(max_len_src, len(ids_src))
        max_len_tar = max(max_len_tar, len(ids_tar))

    print(f'>>> Max length of source sentence: {max_len_src}')
    print(f'>>> Max length of target sentence: {max_len_tar}')

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    return train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tar


def get_model(config, src_vocab_size, tar_vocab_size):
    
    model = build_transformer(src_vocab_size=src_vocab_size, 
                              tar_vocab_size=tar_vocab_size, 
                              src_seq_len=config['seq_len'], 
                              tar_seq_len=config['seq_len'], 
                              dim_model=config['dim_model'])
    return model


def train(config):

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")
    if device == "cuda":
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        total_memory = torch.cuda.get_device_properties(device_index).total_memory / 1024 ** 3 # convert to GB
        print(f">>> Device name: {device_name}")
        print(f">>> Device memory: {total_memory:.2f} GB")
    device = torch.device(device)

    # make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tar = get_dataset(config)

    model = get_model(config=config,
                      src_vocab_size=tokenizer_src.get_vocab_size(),
                      tar_vocab_size=tokenizer_tar.get_vocab_size()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    loss = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), 
                               label_smoothing=0.1).to(device) 
    """
        label_smoothing: to prevent the model from becoming overconfident, improve generalization,
                         reduce overfitting and stabilize training in seq2seq models like Transformers
    """

    # preload a specific model before training, if the user needs to do so
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    
    
