#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path


def get_config():
    
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "learning_rate": 10**-4,
        "seq_len": 350,
        "dim_model": 512,
        "data_source": 'opus_books',
        "lang_src": "en",
        "lang_tar": "it",
        "model_folder": "weights",
        "model_basename": "transformer_model_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transformer_model"
    }


def get_weights_file_path(config, epoch: str):
    
    model_folder = f"{config['data_source']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename) # "./opus_books_weights/transformer_model_epoch.pt"


