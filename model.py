import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tree import *

def apply_masked_logits(target, mask):
    """
    This function applies a mask to a target tensor and adjusts values based on the mask.
    It is typically used in attention mechanisms where certain values need to be ignored.

    Args:
        target (Tensor): The input tensor to which the mask is to be applied.
        mask (Tensor): The mask tensor, same shape as the target.

    Returns:
        Tensor: Masked target tensor.
    """
    return target * mask + (1 - mask) * (-1e30)


class AttentionBasedRelation(nn.Module):
    """
    Attention-based mechanism for determining relations in a sequence.
    Learns the importance of different parts of the input feature sequence
    and returns a weighted sum.

    Args:
        in_dim (int, optional): The dimension of the input feature vectors. Defaults to 300.
        hidden_dim (int, optional): The dimension of the hidden layer. Defaults to 64.
    """
    def __init__(self, in_dim=300, hidden_dim=64):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_features, dependency_tags, mask):
        """
        Forward pass of the Attention-based relation model.
        
        Args:
            input_features (Tensor): The input feature tensor of shape [N, L, D].
            dependency_tags (Tensor): The tensor holding dependency tags of shape [N, L, D].
            mask (Tensor): The tensor for masking invalid values, of shape [N, L].
        
        Returns:
            Tensor: The output tensor after applying attention mechanism.
        """
        attn_weights = self.fc1(dependency_tags)
        attn_weights = self.relu(attn_weights)
        attn_weights = self.fc2(attn_weights).squeeze(2)
        attn_weights = F.softmax(apply_masked_logits(attn_weights, mask), dim=1)

        attn_weights = attn_weights.unsqueeze(2)
        output = torch.bmm(input_features.transpose(1, 2), attn_weights)
        return output.squeeze(2)


class HighwayNetwork(nn.Module):
    """
    A Highway Network layer applies a gating mechanism to its inputs.
    It controls what information should be allowed to flow through the network.

    Args:
        layer_num (int): The number of highway layers.
        dim (int): The dimension of the input and output.
    """
    def __init__(self, layer_num, dim):
        super().__init__()

        self.layer_num = layer_num
        self.linear_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_num)])
        self.gate_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_num)])

    def forward(self, x):
        """
        Forward pass of the Highway Network model.
        
        Args:
            x (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor after applying the highway layers.
        """
        for i in range(self.layer_num):
            gate_output = torch.sigmoid(self.gate_layers[i](x))
            nonlinear_output = torch.relu(self.linear_layers[i](x))
            x = gate_output * nonlinear_output + (1 - gate_output) * x
        return x



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
            self.dependency_highway = HighwayNetwork(config.num_layers, config.embedding_dim)
            self.highway = HighwayNetwork(config.num_layers, config.embedding_dim)

        gcn_input_dim = config.embedding_dim

        # GAT layers for dependency features
        self.gat_dependency = [AttentionBasedRelation(in_dim=config.embedding_dim).to(config.device) for i in range(config.num_heads)]

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