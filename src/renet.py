from pandas.core import base
import torch.nn as nn
import numpy as np
import torch
import dgl
from src.aggregator import MeanAggregator, RGCNAggregator
from rgcn.utils import move_dgl_to_cuda, get_sorted_s_r_embed_rgcn, get_sorted_s_r_embed


class RENet(nn.Module):
    def __init__(self, num_entities, num_relations, time_gap, time_start, hidden_dim, time_dim, renet_dropout_rate, seq_len, phase, encoder_name,
                 seq_model, gpu):
        super(RENet, self).__init__()
        # assert (args.entity_dim == args.relation_dim)
        self.h_dim = hidden_dim
        self.time_start = time_start
        self.t_dim = time_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.renet_dropout_rate = renet_dropout_rate
        self.seq_len = seq_len# for seq
        self.gpu = gpu
        self.phase = phase
        self.seq_model = seq_model
        self.time_gap = time_gap
        self.encoder_name = encoder_name
        # self.node_encoder = None
        # self.sub_encoder = None
        # print(self.seq_model)

        self.dropout = nn.Dropout(self.renet_dropout_rate)
        # self.sub_encoder = nn.GRU(3 * self.h_dim + self.t_dim, self.h_dim, batch_first=True)
        # print(self.phase, "!!!!!!!!!")
        # if self.phase == 1:
        #     self.node_encoder = nn.GRU(self.h_dim + self.t_dim, self.h_dim, batch_first=True)
        # else:
            # self.node_encoder = nn.GRU(self.h_dim + self.t_dim, self.h_dim, batch_first=True)

        if self.phase == 1:
            self.node_encoder = nn.GRU(self.h_dim+self.t_dim, self.h_dim, batch_first=True)

            if self.seq_model == "gru":
                self.sub_encoder = nn.GRU(self.h_dim+self.t_dim, self.h_dim, batch_first=True) 
            else:
                # self.sub_encoder = nn.GRU(3*self.h_dim + self.t_dim, self.h_dim, batch_first=True)  
                # self.node_encoder = nn.GRU(self.h_dim + self.t_dim, self.h_dim, batch_first=True) 
                sub_encoder_layer = nn.TransformerEncoderLayer(d_model=3*self.h_dim+self.t_dim, nhead=4)
                self.sub_encoder = nn.TransformerEncoder(sub_encoder_layer, num_layers=1)
                self.sep_sub = torch.nn.Parameter(torch.Tensor(1, 3*self.h_dim + self.t_dim).float())
                nn.init.xavier_uniform_(self.sep_sub, gain=nn.init.calculate_gain('relu'))
        else:  

            if self.seq_model == 'gru':
                self.node_encoder = nn.GRU(self.h_dim + self.t_dim, self.h_dim, batch_first=True)
                self.sub_encoder = nn.GRU(self.h_dim + self.t_dim, self.h_dim, batch_first=True) 
            else:
                self.sep = torch.nn.Parameter(torch.Tensor(1, self.h_dim + self.t_dim).float())
                nn.init.xavier_uniform_(self.sep, gain=nn.init.calculate_gain('relu'))
                self.sep_sub = torch.nn.Parameter(torch.Tensor(1, 3*self.h_dim + self.t_dim).float())
                nn.init.xavier_uniform_(self.sep_sub, gain=nn.init.calculate_gain('relu'))
                
                encoder_layer = nn.TransformerEncoderLayer(d_model=self.h_dim+self.t_dim, nhead=4)
                self.node_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                # self.linear_sub = nn.Linear(3*self.h_dim+self.t_dim, self.h_dim)
                # nn.init.xavier_normal_(self.linear_sub.weight, gain=nn.init.calculate_gain('relu')) 
                self.linear_node = nn.Linear(self.h_dim+self.t_dim, self.h_dim)
                nn.init.xavier_normal_(self.linear_node.weight, gain=nn.init.calculate_gain('relu')) 

        self.aggregator_s = MeanAggregator(self.h_dim, self.t_dim, self.renet_dropout_rate, self.seq_len)
        # rgcn aggregator 
        self.aggregator_rgcn = RGCNAggregator(self.h_dim, self.t_dim, self.renet_dropout_rate,
                                              self.num_entities, self.num_relations, self.encoder_name, 
                                              phase=self.phase, seq_len=self.seq_len, gpu=self.gpu)

        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.t_dim))).float())
        self.b = torch.nn.Parameter(torch.zeros(self.t_dim).float()) 

        self.criterion = nn.CrossEntropyLoss()
    
    def entity_forward(self, t, cand, packed_input_node, graph_dict, entity_embeddings, relation_embeddings, day_embeddings, phase=1):
        # get the history for all (s, r) pairs (including ducplicate triples)
        # s_hist_len= torch.LongTensor(list(map(len, history[0]))).cuda()
        # s_len, s_idx= s_hist_len.sort(0, descending=True)
        
        if phase == 1:
            g_list, node_ids_graph, s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, s_time_week, padding_mask, s_len, s_idx = packed_input_node
        else:
            history = packed_input_node
            g_list, node_ids_graph, s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, s_time_week, padding_mask, s_len, s_idx = get_sorted_s_r_embed_rgcn((history[0], history[1]), cand, t, graph_dict, self.seq_len, self.time_gap, self.time_start)
        padding_mask_all = torch.ones(len(s_len), self.seq_len).cuda()
        if s_len[0] > 0 :
            # padding_mask = padding_mask.cuda()
            g_list = [g.to(torch.device('cuda:'+str(torch.cuda.current_device()))) for g in g_list]  
            batched_graph = dgl.batch(g_list)
            batched_graph.ndata['h'] = entity_embeddings[batched_graph.ndata['id']].view(-1, entity_embeddings.shape[1])
            move_dgl_to_cuda(batched_graph)
            s_embed_seq_tensor = self.aggregator_rgcn.forward(cand,
                                                          batched_graph,
                                                          entity_embeddings,
                                                          relation_embeddings.data, 
                                                          day_embeddings,
                                                          self.w, 
                                                          self.b,
                                                          s_len_non_zero,
                                                          s_time_iterval_sorted,
                                                          s_time_day,
                                                          s_time_week,
                                                          node_ids_graph)
            if self.seq_model == "gru":
                s_embed_seq_tensor = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                        s_len_non_zero.cpu(),
                                                                        batch_first=True)   
                # _, s_h = self.node_encoder(s_embed_seq_tensor)
                output, s_h = self.node_encoder(s_embed_seq_tensor)
                s_h_1 = s_h.squeeze(0)
                s_h_1 = torch.cat((s_h_1, torch.zeros(len(cand) - len(s_h_1), self.h_dim).cuda()), dim=0)
                if self.phase == 2:
                    s_h_2, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                    if s_len[0] < self.seq_len:
                        s_h_padding = torch.zeros(s_h_2.size(0), self.seq_len-s_h_2.size(1), s_h_2.size(2)).cuda()
                        s_h_2 = torch.cat((s_h_2, s_h_padding), dim=1) 
                    s_h_2 = torch.cat((s_h_2, torch.zeros(len(cand)-s_h_2.size(0), self.seq_len, self.h_dim).cuda())) 
            else:
                sep_input = self.sep.expand(s_embed_seq_tensor.size(0), -1).unsqueeze(1)
                # print(sep_input.size(), s_embed_seq_tensor.size())
                s_embed_seq_tensor = torch.cat([sep_input, s_embed_seq_tensor], dim = 1)

                s_h = self.node_encoder.forward(s_embed_seq_tensor.transpose(0, 1), src_key_padding_mask=padding_mask.bool())
                # s_h = s_h[0, :, :].squeeze(0)
                s_h = self.linear_node(s_h)
                s_h = s_h.transpose(0, 1)
                # print(s_h.size())
                s_h_1 = s_h[:, 0, :].squeeze(0)
                s_h_1 = torch.cat((s_h_1, torch.zeros(len(cand) - len(s_h_1), self.h_dim).cuda()), dim=0)
                if phase == 2:
                    s_h_2 = s_h[:, 1:, :]
                # # print(s_h.size())
                # if s_len[0] < self.seq_len:
                    # s_h_padding = torch.zeros(s_h_2.size(0), self.seq_len-s_h_2.size(1), s_h_2.size(2)).cuda()
                    # s_h_2 = torch.cat((s_h_2, s_h_padding), dim=1)
                    # print(s_h_2.size())
                    s_h_2 = torch.cat((s_h_2, torch.zeros(len(cand)-s_h_2.size(0), self.seq_len, self.h_dim).cuda()))
                    
                # s_h_2 = torch.cat((s_embed_seq_tensor, torch.zeros(len(cand) - s_embed_seq_tensor.size(0), self.seq_len, self.h_dim+self.t_dim).cuda()), dim=0)
        else:
            s_h_1 = torch.zeros(len(cand), self.h_dim).cuda()
            if phase == 2:
                s_h_2 = torch.zeros(len(cand), self.seq_len, self.h_dim).cuda()
        # print(s_h.size())
        # restitute the node order to the original order
        padding_mask_all[:padding_mask.size(0), :] = padding_mask[:, 1:]
        inverse_s_idx = torch.zeros(len(s_idx)).long()
        for i, s in enumerate(s_idx.tolist()):
            inverse_s_idx[s] = i
        s_h_1 = s_h_1[inverse_s_idx]
        # print(s_h_1.size())
        if phase == 2:
            # print(padding_mask_all.size(), len(inverse_s_idx))
            s_h_2 = s_h_2[inverse_s_idx]
            padding_mask_all = padding_mask_all[inverse_s_idx, :]
        # get all triples' [s, r] embeddings
        # print(s.size(), s_h.size())

        # short_entity = s_h[all_triples[:, 0]]
        # short_relation = relation_embeddings[all_triples[:, 1]]
        # print(short_entity.size(), short_relation.size())
        # e1_embedded = short_entity.unsqueeze(1)
        # rel_embedded = short_relation.unsqueeze(1)
        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        # if s_len[0] > 0:
        #     print(s_h.size(), s_hist_len)
        if phase == 1:
            return s_h_1
        else:
            return s_h_1, s_h_2, padding_mask_all.bool()

    def forward(self, t, all_triples, packed_input_snap, entity_embeddings, relation_embeddings, day_embeddings, phase=1):
        '''
        :param triplets:list with each element :[head, tail, rel, time]
        :param s_hist:  history of each element : list of list [[tail_t1_0, tail_t1_1],[tail_t2_1],[tail_t3_1, tail_t3_2],[]..,[]]
        :param kg:  object of knowledge graph class kg.graph_dict = None if model = 1,2
        :return:
        '''
        # print(all_triples.size())
        s, r= all_triples[:, 0], all_triples[:, 1]
        # s_hist_len = torch.LongTensor(list(map(len, hist[0]))).cuda()
        # print(s_hist_len)
        # s_len, s_idx = s_hist_len.sort(0, descending=True)
        if phase == 1:
            s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, s_time_week, padding_mask, neigh_num_of_s, flat_s, s_len_sorted, s_idx = packed_input_snap
        else:
            history = packed_input_snap
            # s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, s_time_week, padding_mask, neigh_num_of_s, flat_s, s_len_sorted, s_idx
            s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, s_time_week, padding_mask, neigh_num_of_s, flat_s, s_len_sorted, s_idx = get_sorted_s_r_embed((history[0], history[1]), s, r, t, self.seq_len, self.time_gap, self.time_start)
        # print("!", len(s_idx))
        if s_len_sorted[0] > 0:
            s_embed_seq_tensor = self.aggregator_s.forward(s, r, s_idx, entity_embeddings, relation_embeddings, day_embeddings,
                                    self.w, self.b, flat_s, neigh_num_of_s, s_len_non_zero, s_time_iterval_sorted, s_time_day, s_time_week)
            if self.seq_model == "gru":
                s_embed_seq_tensor = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                        s_len_non_zero.cpu(),
                                                        batch_first=True)  
                # print("!!!", s_embed_seq_tensor.size())
                _, s_h = self.sub_encoder(s_embed_seq_tensor)
                s_h = s_h.squeeze(0)
            else:
                sep_sub = self.sep_sub.expand(s_embed_seq_tensor.size(0), -1).unsqueeze(1)
                s_embed_seq_tensor = torch.cat([sep_sub, s_embed_seq_tensor], dim=1)
                s_h = self.sub_encoder.forward(s_embed_seq_tensor.transpose(0, 1), src_key_padding_mask=padding_mask.bool())
                s_h = self.linear_sub(s_h)
                s_h = s_h.transpose(0, 1)
                s_h = s_h[:, 0, :].squeeze(0)
            s_h = torch.cat((s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()), dim=0)
            # print("s_h", s_h.size())
        else:
            s_h = torch.zeros(len(s) , self.h_dim).cuda()

        inverse_s_idx = torch.zeros(len(s_idx)).long()
        for i, s in enumerate(s_idx.tolist()):
            inverse_s_idx[s] = i
        # ob_pred_inverse = ob_pred[inverse_s_idx]
        local_hidden =  s_h[inverse_s_idx]
        return local_hidden

