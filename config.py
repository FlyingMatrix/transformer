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
        "lang_tar": "it"
    }

