# coding=utf-8
import argparse
import logging
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random

import numpy as np
import pandas as pd
import torch
from transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)
from torch.utils.data import DataLoader

from datasets import load_datasets_and_vocabs
from model import Aspect_Bert_GAT
from trainer import train_model

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def check_args(args):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(args))
        


class ArgumentParser:
    def __init__(self):
        pass


args = ArgumentParser()

# Required parameters
args.dataset_name = 'rest'
args.output_dir = '/data/output-gcn'
args.num_classes = 3
args.cuda_id = '3'
args.seed = 2019

# Model parameters
args.glove_dir = '/data1/SHENWZH/wordvec'
args.bert_model_dir = 'bert-base-uncased'
args.pure_bert = False
args.gat_bert = False
args.highway = False
args.num_layers = 2
args.add_non_connect = True
args.multi_hop = True
args.max_hop = 4
args.num_heads = 6
args.dropout = 0.3
args.num_gcn_layers = 1
args.gcn_mem_dim = 300
args.gcn_dropout = 0.2
args.gat = False
args.gat_our = False
args.gat_attention_type = 'dotprod'
args.embedding_type = 'bert'
args.embedding_dim = 300
args.dep_relation_embed_dim = 300
args.hidden_size = 200
args.final_hidden_size = 300
args.num_mlps = 2

# Training parameters
args.per_gpu_train_batch_size = 16
args.per_gpu_eval_batch_size = 32
args.gradient_accumulation_steps = 2
args.learning_rate = 5e-5
args.weight_decay = 0.0
args.adam_epsilon = 1e-8
args.max_grad_norm = 1.0
args.num_train_epochs = 30.0
args.max_steps = -1
args.logging_steps = 50


# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# Parse args
check_args(args)

# Setup CUDA, GPU training
#os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
args.device = device
logger.info('Device is %s', args.device)

# Set seed
set_seed(args)

# Bert, load pretrained model and tokenizer, check if neccesary to put bert here
#if args.embedding_type == 'bert':
tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
args.tokenizer = tokenizer

# Load datasets and vocabs
train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab= load_datasets_and_vocabs(args)

# Build Model

model = Aspect_Bert_GAT(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # R-GAT + Bert

model.to(args.device)
# Train
_, _,  all_eval_results = train_model(args, train_dataset, model, test_dataset)

if len(all_eval_results):
    best_eval_result = max(all_eval_results, key=lambda x: x['acc']) 
    for key in sorted(best_eval_result.keys()):
        logger.info("  %s = %s", key, str(best_eval_result[key]))

