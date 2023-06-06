import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_gcn import GAT, GCN, Rel_GAT
from model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits
from tree import *


class Aspect_Bert_GAT(nn.Module):
    '''
    R-GAT with bert
    '''

    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Bert_GAT, self).__init__()
        self.args = args

        # Bert
        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf =False)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)


        gcn_input_dim = args.embedding_dim

        # GAT
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.num_heads)]


        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer,input_cat_ids,segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
        fmask[:,0] = 1
        outputs = self.bert(input_cat_ids, token_type_ids = segment_ids)
        feature_output = outputs[0] # (N, L, D)
        pool_out = outputs[1] #(N, D)

        # index select, back to original batched size.
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_indexer)])

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        dep_out = dep_out.mean(dim=1)  # (N, D)


        feature_out = torch.cat([dep_out,  pool_out], dim=1)  # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
