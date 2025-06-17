#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, get_latest_weights_file

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

from torch.utils.tensorboard import SummaryWriter


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


# greedy_decode: select the next token with the highest probability
def greedy_decode(model, src, encoder_mask, tokenizer_src, tokenizer_tar, max_len, device): 
    
    sos_idx = tokenizer_tar.token_to_id("[SOS]") # start of sentence
    eos_idx = tokenizer_tar.token_to_id("[EOS]") # end of sentence

    # compute the encoder output
    encoder_output = model.encode(src, encoder_mask)
    # initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build decoder_mask based on the current decoder_input
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
        # calculate the output 
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        project_output = model.project(decoder_output) # project_output -> (batch_size=1, seq_len, vocab_size)

        # get next token
        prob = project_output[:, -1] # prob -> (batch_size=1, vocab_size)
        _, next_token = torch.max(prob, dim=1)
        # concatenate next_token with decoder_input
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).fill_(next_token.item()).type_as(src).to(device)], dim=1
        )

        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)


def validation(model, valid_dataloader, tokenizer_src, tokenizer_tar, max_len, device, 
               print_msg, global_step, writer, num_examples=2):
    
    model.eval()
    counter = 0
    src_texts = []
    predicted = []
    expected = []

    # get the console window width
    try:
        with os.popen('stty size', 'r') as console:
            console_height, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # if getting console width failed, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in valid_dataloader:
            counter += 1

            encoder_input = batch['encoder_input'].to(device) # encoder_input -> (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # encoder_mask -> (batch_size, 1, 1, seq_len)

            # check if the batch_size for validation is 1
            assert encoder_input.size(0) == 1, "batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tar, max_len, device) # model_output is token IDs

            source_text = batch["src_sentence"][0]
            target_text = batch["tar_sentence"][0]
            model_output_text = tokenizer_tar.decode(model_output.detach().cpu().numpy()) # token IDs -> string

            src_texts.append(source_text)
            predicted.append(model_output_text)
            expected.append(target_text)

            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_output_text}")

            if counter == num_examples:
                print_msg('-' * console_width)
                break

        if writer:
            pass        






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

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # preload a specific model before training, if the user needs to do so
    initial_epoch = 0 # used to resume training from a particular epoch
    global_step = 0 # used in logging, learning rate scheduling, or saving checkpoints
    preload = config['preload'] 
    model_filename = (
        get_latest_weights_file(config) if preload == "latest"
        else get_weights_file_path(config, preload) if preload != None
        else None        
    )
    if model_filename:
        print(f'>>> Preloading model: {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer_state_dict'])
    else:
        print('>>> No model to preload...')

    # training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f'>>> Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # encoder_input -> (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # decoder_input -> (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # encoder_mask -> (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # decoder_mask -> (batch_size, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) 
            # encoder_output -> (batch_size, seq_len, dim_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            # decoder_output -> (batch_size, seq_len, dim_model)
            project_output = model.project(decoder_output)
            # project_output -> (batch_size, seq_len, vocab_size)

            # get labels
            labels = batch['decoder_label'].to(device) # labels -> (batch_size, seq_len)

            # compute the loss
            output = project_output.view(-1, tokenizer_tar.get_vocab_size()) 
            # output -> (batch_size * seq_len, tar_vocab_size) -> (N, C)
            labels = labels.view(-1) # labels -> (batch_size * seq_len) -> (N)
            loss = loss(output, labels)

            # log the loss to tensorboard
            writer.add_scalar('train_loss', loss.item(), global_step)
            # ensure that all scalar values, histograms, images, etc., are written out to the log files
            writer.flush() 

            # backpropagate the loss and update the weights
            optimizer.zero_grad() # clear old gradients
            loss.backward() # backward pass - computes gradients
            optimizer.step() # update weights

            # update the global_step
            global_step += 1

        # run validation after each epoch
        validation()

        # save the model after each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step
                   }, 
                   model_filename)      


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    config = get_config()
    train(config)


