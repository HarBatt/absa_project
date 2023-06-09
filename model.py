import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_utils import RelationAttention, Highway
from tree import *


class BertGraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network with BERT embeddings.
    """

    def __init__(self, config, dependency_tag_count, pos_tag_count):
        """
        Initialize the network.
        """
        super(BertGraphAttentionNetwork, self).__init__()
        self.config = config

        # Bert initialization
        bert_config = BertConfig.from_pretrained(config.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            config.bert_model_dir, config=bert_config, from_tf =False)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.dropout)
        config.embedding_dim = bert_config.hidden_size

        # Highway layers for dependency and normal features
        if config.highway:
            self.dependency_highway = Highway(config.num_layers, config.embedding_dim)
            self.highway = Highway(config.num_layers, config.embedding_dim)

        gcn_input_dim = config.embedding_dim

        # GAT layers for dependency features
        self.gat_dependency = [RelationAttention(in_dim=config.embedding_dim).to(config.device) for i in range(config.num_heads)]

        self.dep_embedding = nn.Embedding(dependency_tag_count, config.embedding_dim)

        last_hidden_size = config.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, config.final_hidden_size), nn.ReLU()]
        for _ in range(config.num_mlps - 1):
            layers += [nn.Linear(config.final_hidden_size,
                                 config.final_hidden_size), nn.ReLU()]
        self.final_layers = nn.Sequential(*layers)
        self.final_output_layer = nn.Linear(config.final_hidden_size, config.num_classes)

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer,input_cat_ids,segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        """
        Forward pass of the network.
        """
        # Create mask for feature computation
        feature_mask = (torch.ones_like(word_indexer) != word_indexer).float() 
        feature_mask[:,0] = 1

        # Get outputs from BERT
        bert_outputs = self.bert(input_cat_ids, token_type_ids = segment_ids)
        feature_output = bert_outputs[0] 
        pool_output = bert_outputs[1] 

        # Retrieve original batched size
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_indexer)])

        # Compute dependency features
        dep_feature = self.dep_embedding(dep_tags)
        if self.config.highway:
            dep_feature = self.dependency_highway(dep_feature)

        # Apply GAT on dependency features
        dep_out = [g(feature, dep_feature, feature_mask).unsqueeze(1) for g in self.gat_dependency]
        dep_out = torch.cat(dep_out, dim=1)
        dep_out = dep_out.mean(dim=1)

        # Concatenate outputs
        feature_out = torch.cat([dep_out,  pool_output], dim=1)

        # Apply dropout and final layers
        x = self.dropout(feature_out)
        x = self.final_layers(x)
        logit = self.final_output_layer(x)
        return logit


def initialize_zero_rnn_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    """
    Create a zero-state for RNN.
    """
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0