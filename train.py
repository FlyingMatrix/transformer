#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Huggingface tokenizers
"""
    Huggingface Tokenizer best for: Transformer-based models (BERT, GPT, etc.)
"""
import torchtext.datasets as datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


