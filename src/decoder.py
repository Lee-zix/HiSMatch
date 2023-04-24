from html import entities
import random
from xml.sax import xmlreader

from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os
path_dir = os.getcwd()


class ConvEvolve(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, t_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvEvolve, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        # self.loss = torch.nn.BCELoss()
        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear((embedding_dim) * channels, embedding_dim)

        
    def forward(self, global_e, global_r, head_h , query_h, evolve_embs, triplets, phase=1):
        batch_size = len(triplets)
        evolve_embs = F.tanh(evolve_embs)
        # e1_embedded = global_e[triplets[:, 0]]
        rel_embedded = global_r[triplets[:, 1]]
        hidden_embedded = head_h + query_h
        # hidden_embedded = query_h
        # hidden_embedded = head_h
        # hidden_embedded = query_h
        # print(rel_embedded.size(), day_h.size())
        # rel_embedded = torch.cat([rel_embedded, day_h], dim=1)
        # hidden_embedded = torch.cat([hidden_embedded, day_h], dim=1)
        # hidden_embedded = head_h + rel_embedded
        # hidden_embedded = e1_embedded + head_h
        # print(hidden_embedded)
        stacked_inputs = torch.cat([hidden_embedded.unsqueeze(1), rel_embedded.unsqueeze(1)], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)

        # query reshaping
        # query_inputs = x.view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        # entity_inputs = evolve_embs.view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        # query_x = self.bn4(query_inputs)
        # query_x = self.conv2(query_x)
        # query_x = F.relu(query_x)
        # query_x = self.feature_map_drop(query_x)
        # query_x = query_x.view(-1, self.feat_dim)
        # query_x = self.fc_2(query_x)
        # query_x = self.hidden_drop(query_x)
        # query_x = self.bn6(query_x)
        # query_x = F.relu(query_x)

        # entity_x = self.bn4(entity_inputs)
        # entity_x = self.conv2(entity_x)
        # entity_x = F.relu(entity_x)
        # entity_x = self.feature_map_drop(entity_x)
        # entity_x = entity_x.view(-1, self.feat_dim)
        # entity_x = self.fc_2(entity_x)
        # entity_x = self.hidden_drop(entity_x)
        # entity_x = self.bn6(entity_x)
        # entity_x = F.tanh(entity_x)
        
        x = F.relu(x)
        # print(x.size(), evolve_embs.size())
        # evolve_embs = evolve_embs + global_e 
        if phase == 1:
            x = torch.mm(x, evolve_embs.transpose(1, 0))
        elif phase == 2:
            # print(x.unsqueeze(1).size())
            # print(evolve_embs.transpose(1,0).transpose(1,2).size())
            # print(x.unsqueeze(1))
            x = torch.matmul(x.unsqueeze(1), evolve_embs.transpose(1,0).transpose(1,2))
            x = x.squeeze(1)
        return x
