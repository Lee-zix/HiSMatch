import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
# from dgl.nn.pytorch.glob import AvgPooling
# from src.reasoner.utils import *
# from RGCN import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, UnionRGATLayer
# from src.utils.ops import var_cuda
from rgcn.utils import get_sorted_s_r_embed, get_sorted_s_r_embed_rgcn
import time

# class PositionalEncoding(nn.Module):
#     "Implement the PE function"
#     def __init__(self, d_model, dropout, max_len=10):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         # Compute the position encoding once in log space
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *- (math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # ic(self.pe.size(), x.size()) # [1, 169,1024]  [:, :64]  --> [169,1,1024]
#         # ic(self.pe[:, :x.size(1)].size())
#         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#         # return self.dropout(x)
#         return x


class MeanAggregator(nn.Module):
    def __init__(self, h_dim, t_dim, dropout, seq_len=10):
        super(MeanAggregator, self).__init__()
        self.h_dim = h_dim
        self.t_dim = t_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        # self.padding_vec = Variable(torch.zeros(1, self.h_dim), requires_grad=False).cuda()
  
    def forward(self, s, r, s_idx, ent_embeds, rel_embeds, day_embeds, t_w, t_b, flat_s, len_s, s_len_non_zero, s_time_iterval_sorted, s_time_day, s_time_week):
        # if len(s_hist) == 2:
        #     s_hist = s_hist[0]
        # s_len_non_zero, s_tem, s_time_iterval_sorted, s_time_month, s_time_day, r_tem, len_s, flat_s = get_sorted_s_r_embed(hist, s, r, t)
        # To get mean vector at each time
        s_tem = s[s_idx]
        r_tem = r[s_idx]
        embeds_stack = ent_embeds[torch.LongTensor(flat_s).cuda()]
        embeds_split = torch.split(embeds_stack, len_s)   # Splits the tensor into chunks according to len_s

        curr = 0
        rows = []
        cols = []
        for i, leng in enumerate(len_s): # s_len length of each subgraph of each example: [3, 2, 1, 1, 2]    s_len_non_zero : subgraph number of each example [3 , 2]
            rows.extend([i] * leng)     # [0, 0, 0, 1, 1, 2, 3, 4, 4]
            cols.extend(list(range(curr,curr+leng)))    # [0, 1, 2, 3, 4, 5, 6, 7, 8]
            curr += leng
        rows = torch.LongTensor(rows)
        cols = torch.LongTensor(cols)
        idxes = torch.stack([rows,cols], dim=0)

        # torch 0.4.1(一种声明稀疏矩阵的方法)
        # >>> i = torch.LongTensor([[2, 4]])
        # >>> v = torch.FloatTensor([[1, 3], [5, 7]])
        # >>> torch.sparse.FloatTensor(i, v).to_dense()
        #  0  0
        #  0  0
        #  1  3
        #  0  0
        #  5  7
        mask_tensor = torch.sparse.FloatTensor(idxes, torch.ones(len(rows))) #
        mask_tensor = mask_tensor.cuda()
        # embeds_sum [len of sub graph, dim ] 一个subgraph 表示一个样本在某一个时刻通过r连接的所有实体
        embeds_sum = torch.sparse.mm(mask_tensor, embeds_stack)
        embeds_mean = embeds_sum /torch.Tensor(len_s).cuda().view(-1,1)

        embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist()) # s_len_non_zero : subgraph number of each example
    
        '''
        [exampleNum, seq_len, 3*h_dim]: 3-D matrix as the input of RNN
        for i, embeds in enumerate(embeds_split) give value to corresponding subgraph, others podding with zeros
        '''

        # week_embed_seq_tensor=torch.ones(len(s_len_non_zero), self.seq_len).cuda()*7
        # day_embed_seq_tensor=torch.ones(len(s_len_non_zero), self.seq_len).cuda()*31
        time_embed_seq_tensor=torch.ones(len(s_len_non_zero), self.seq_len).cuda()*1000000
        # print(len(time_embed_seq_tensor), len(s_time_iterval_sorted))

        for i, (time_snap, day_snap, week_snap) in enumerate(zip(s_time_iterval_sorted, s_time_day, s_time_week)):
            # print(time_snap)
            time_embed_seq_tensor[i, torch.arange(len(time_snap))] = time_snap.cuda()
            # week_embed_seq_tensor[i, torch.arange(len(week_snap))] = week_snap.cuda()
            # day_embed_seq_tensor[i, torch.arange(len(day_snap))] = day_snap.cuda()
        # print("time_embed_seq_tensor is {}".format(time_embed_seq_tensor))
        time_embed_seq_tensor=time_embed_seq_tensor.long()
        # week_embed_seq_tensor = week_embed_seq_tensor.long()
        # day_embed_seq_tensor = day_embed_seq_tensor.long()

        time_embed_seq_tensor = time_embed_seq_tensor.view(-1, 1)
        time_embed_seq_tensor = torch.cos(time_embed_seq_tensor * t_w + t_b)
        time_embed_seq_tensor = time_embed_seq_tensor.view(-1, self.seq_len, self.t_dim)

        # week_embed_seq_tensor = week_embed_seq_tensor.view(-1, 1)
        # week_embed_seq_tensor = day_embeds[week_embed_seq_tensor]
        # week_embed_seq_tensor = week_embed_seq_tensor.view(-1, self.seq_len, self.t_dim)

        # day_embed_seq_tensor = day_embed_seq_tensor.view(-1, 1)
        # day_embed_seq_tensor = day_embeds[day_embed_seq_tensor]
        # day_embed_seq_tensor = day_embed_seq_tensor.view(-1, self.seq_len, self.t_dim)
        
        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, self.h_dim).cuda()
        # Slow!!!
        # for i, embeds in enumerate(embeds_split):
        #     s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
        #         (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1),
        #         rel_embeds[r_tem[i]].repeat(len(embeds), 1)), dim=1)
        for i, embeds in enumerate(embeds_split):
            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = embeds

        s_embed_seq_tensor=torch.cat((s_embed_seq_tensor, time_embed_seq_tensor), dim=-1)#, week_embed_seq_tensor, day_embed_seq_tensor),dim=-1)      
        # 
        # s_embed_seq_tensor=torch.cat((s_embed_seq_tensor, time_embed_seq_tensor, week_embed_seq_tensor, day_embed_seq_tensor), dim=-1)
        # print(s_embed_seq_tensor.size(), time_embed_seq_tensor.size() )
        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
            # Packs a Tensor containing padded sequences of variable length
            # Input: Input can be of size T*B*? where T is the length of the longest sequence
            #                                         B is the batch size
            #                                         * is any number of dimensions (including 0).
            #        B*T*? inputs are expected if batch_first is True!!!!!!
            #        The sequences should be sorted by length in a decreasing order,
            #        i.e. input[:,0] should be the longest sequence, and input[:,B-1] the shortest one.
            # print(s_embed_seq_tensor.size())
        # print("s_len_non_zero", s_len_non_zero)
        # s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
        #                                                          s_len_non_zero.cpu(),
        #                                                          batch_first=True)

        # return s_packed_input
        return s_embed_seq_tensor



class RGCNAggregator(nn.Module):
    def __init__(self, h_dim, t_dim, dropout, num_nodes, num_rels, encoder_name, phase=1, seq_len=10, gpu=0):
        super(RGCNAggregator, self).__init__()
        self.h_dim = h_dim
        self.t_dim = t_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.gpu = gpu
        self.encoder_name = encoder_name

        if self.encoder_name.startswith("uvrgcn"):
            comp = encoder_name.split("-")[1]
            self.rgcn1 = UnionRGCNLayer(self.h_dim, self.h_dim, 2*self.num_rels, comp,
                            activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)
            self.rgcn2 = UnionRGCNLayer(self.h_dim, self.h_dim, 2*self.num_rels, comp,
                               activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)
        elif self.encoder_name == "rgcn":
            self.rgcn1 = RGCNBlockLayer(self.h_dim, self.h_dim, 2*self.num_rels, num_bases=32,
                            activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)
            self.rgcn2 = RGCNBlockLayer(self.h_dim, self.h_dim, 2*self.num_rels, num_bases=32,
                               activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)

        elif self.encoder_name == "kbat":
            self.rgcn1 = UnionRGATLayer(self.h_dim, self.h_dim, 2*self.num_rels, num_bases=32,
                            activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)
            self.rgcn2 = UnionRGATLayer(self.h_dim, self.h_dim, 2*self.num_rels, num_bases=32,
                               activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)
  

        # print(self.gpu) 
        self.padding_vec = torch.zeros(1, self.h_dim).cuda(self.gpu)
        print("uvrgcn")

    # def src_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float(1.0)).masked_fill(mask == 1, float(0.0))
    #     return mask

    def forward(self, s, g, ent_embeds, rel_embeds, day_embeds, t_w, t_b, s_len_non_zero, s_time_iterval_sorted, s_time_day, s_time_week, node_ids_graph):
        # s_len_non_zero, s_tem, s_time_iterval_sorted, g, node_ids_graph, len_s, max_len, num_non_zero, padding_mask = \
        #     get_sorted_s_r_embed_rgcn(s_hist, s, t, ent_embeds, graph_dict, self.seq_len)

        if g is None:
            s_embed_seq_tensor = None
        else: 
            # week_embed_seq_tensor=torch.ones(len(s_len_non_zero), self.seq_len).cuda()*7
            # day_embed_seq_tensor=torch.ones(len(s_len_non_zero), self.seq_len).cuda()*31
            time_embed_seq_tensor=torch.ones(len(s_len_non_zero), self.seq_len).cuda()*1000000

            for i, (time_snap, dap_snap, week_snap) in enumerate(zip(s_time_iterval_sorted, s_time_day, s_time_week)):
                time_embed_seq_tensor[i, torch.arange(len(time_snap))] = time_snap.cuda()
                # week_embed_seq_tensor[i, torch.arange(len(week_snap))] = week_snap.cuda()
                # day_embed_seq_tensor[i, torch.arange(len(dap_snap))] = dap_snap.cuda()
            # print("time_embed_seq_tensor is {}".format(time_embed_seq_tensor))
            time_embed_seq_tensor=time_embed_seq_tensor.long()
            # self.rgcn1(g)
            # self.rgcn2(g)
            if self.encoder_name.startswith("uvrgcn") or self.encoder_name == "kbat":
                self.rgcn1(g, [], rel_embeds)
                self.rgcn2(g, [], rel_embeds)
            elif self.encoder_name == "rgcn":
                self.rgcn1(g, [])
                self.rgcn2(g, [])
                

            embeds = g.ndata.pop('h')
            embeds = torch.cat([embeds, self.padding_vec], dim=0)
            
            s_embed_seq_tensor = embeds[node_ids_graph]
            # print(s_len_non_zero.tolist())
            # print(embeds_mean.size())
            # embeds_split = torch.split(embeds_mean, max_len)

            # graph_embeds = [_.unsqueeze(0) for _ in embeds_split]
            # graph_embeds = torch.cat(graph_embeds, dim=0)
            # print(graph_embeds)

            # s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, self.h_dim).cuda()
            
            # # att_mask = self.src_mask(self.seq_len+1)  
        
            # for i, embeds in enumerate(embeds_split):
            #     s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = embeds

            time_embed_seq_tensor = time_embed_seq_tensor.view(-1, 1)
            time_embed_seq_tensor = torch.cos(time_embed_seq_tensor * t_w + t_b)
            time_embed_seq_tensor = time_embed_seq_tensor.view(-1, self.seq_len, self.t_dim)

            # week_embed_seq_tensor = week_embed_seq_tensor.long()
            # # day_embed_seq_tensor = day_embed_seq_tensor.long()    
            # week_embed_seq_tensor = week_embed_seq_tensor.view(-1, 1)
            # week_embed_seq_tensor = day_embeds[week_embed_seq_tensor]
            # week_embed_seq_tensor = week_embed_seq_tensor.view(-1, self.seq_len, self.t_dim)

            # day_embed_seq_tensor = day_embed_seq_tensor.view(-1, 1)
            # day_embed_seq_tensor = day_embeds[day_embed_seq_tensor]
            # day_embed_seq_tensor = day_embed_seq_tensor.view(-1, self.seq_len, self.t_dim)
        
            # print(s_embed_seq_tensor.size(), time_embed_seq_tensor.size())
            # s_embed_seq_tensor=torch.cat((s_embed_seq_tensor, time_embed_seq_tensor, week_embed_seq_tensor, day_embed_seq_tensor),dim=-1)
            # print(s_embed_seq_tensor.size(), time_embed_seq_tensor.size())
            
            
            
            s_embed_seq_tensor=torch.cat((s_embed_seq_tensor, time_embed_seq_tensor), dim=-1)


            # s_embed_seq_tensor = self.pos_emb_layer(s_embed_seq_tensor)
            s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
            
        return s_embed_seq_tensor
