import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.renet import RENet
from src.decoder import ConvTransE, ConvTransE_plus, ConvTransR, ConvATT, ConvE, ConvTransEpp


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, method=1, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.hidden_dropout = hidden_dropout
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.method = method
        self.gpu = gpu

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.emb_ent = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.emb_ent)

        # self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.w1)

        # self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.w2)

        # self.linear_long = nn.Linear(4 * self.h_dim, num_ents)

        # self.linear_hidden = nn.Linear(4 * self.h_dim, 2 * self.h_dim)

        # # embeddings for local history
        # self.emb_rel_renet = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.emb_rel_renet)
        # self.emb_ent_renet = torch.nn.Parameter(torch.Tensor(self.num_ents, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_uniform_(self.emb_ent_renet)
        # self.emb_rel_renet = self.emb_rel
        # self.emb_ent_renet = self.dynamic_emb

        # RENet for local history
        self.renet = RENet( self.num_ents, 
                            self.num_rels, 
                            self.h_dim, 
                            self.hidden_dropout, 
                            seq_len=10, 
                            method=1,
                            gpu=self.gpu)

        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)                     

        # GRU cell for relation evolving
        # self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        # print(self.method)
        # decoder
        # if self.method == 0 or self.method == 1:
        #     self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        #     self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        # elif self.method == 2 or self.method == 3 or self.method == 5:
        #     self.decoder_ob = ConvATT(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        # elif self.method == 6:
        #     self.decoder_ob = ConvE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        # # else self.method == 3:
        # #     self.decoder_ob = ConvE(num_ents, 2*h_dim, input_dropout, hidden_dropout, feat_dropout)

        # elif self.method == 7:
        self.decoder_ob = ConvTransE_plus(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        # elif self.method == 8:
        #     self.decoder_ob = ConvTransEpp(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout) 

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        self.h = F.normalize(self.emb_ent) if self.layer_norm else self.emb_ent

        history_embs = [self.h]

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            current_h = self.rgcn.forward(g, self.h, [self.emb_rel.data, self.emb_rel.data])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            history_embs.append(self.h)
        return history_embs, self.emb_rel, gate_list, degree_list


    def predict(self, long_term_his, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            test_triplets = torch.cat((test_triplets, inverse_test_triplets))
            # print(len(test_triplets))
            
            evolve_embs, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            
            if self.method > 0:
                long_history = long_term_his[0] + long_term_his[1]
                long_term_h = self.renet.forward(test_triplets, long_history, self.emb_ent, self.emb_rel)    #[B, 2*h_dim]
            else:
                long_term_h = None
            # if self.method == 0:
            #     score = self.decoder_ob.forward(embedding, r_emb, test_triplets, mode="test")

            # elif self.method == 1:
            #     scores_ent0 = self.decoder_ob.forward(embedding, r_emb, test_triplets, mode="test")
            #     scores_ent1 =  self.linear_long(torch.cat((self.emb_ent_renet[test_triplets[:, 0]], long_term_h,
            #                             self.emb_rel_renet[test_triplets[:, 1]]), dim=1))
            #     # score = torch.softmax(scores_ent0, dim=1) + torch.softmax(scores_ent1, dim=1)
            #     score = scores_ent0 + scores_ent1
            # elif self.method == 2 :
            #     score = self.decoder_ob.forward(embedding, r_emb, long_term_h, test_triplets)
            
            # elif self.method == 3:
            #     long_term_h = self.linear_hidden(
            #             torch.cat((self.emb_ent_renet[test_triplets[:, 0]], long_term_h,
            #                                 self.emb_rel_renet[test_triplets[:, 1]]), dim=1))
            #     score = self.decoder_ob.forward(embedding, r_emb, long_term_h, test_triplets) 
            # elif self.method == 4:
            #     score = self.decoder_ob.forward_v1(embedding, r_emb, long_term_h, test_triplets, self.emb_ent_renet, self.emb_rel_renet)

            # elif self.method == 5:
            #     score = self.decoder_ob.forward_v2(embedding, r_emb, long_term_h, test_triplets) 
            
            # elif self.method == 6:
            #     score = self.decoder_ob.forward(embedding, r_emb, long_term_h, test_triplets)
             
            # elif self.method == 7:
            score = self.decoder_ob.forward(embedding, r_emb, long_term_h, test_triplets)
            # elif self.method == 8:
            #     score = self.decoder_ob.forward(embedding, r_emb, long_term_h, test_triplets)

            score_rel = torch.zeros_like(score)
            # print(score[:10,:100])
            return test_triplets, score, score_rel


    def get_loss(self, glist, ans_dict, triples, static_graph, long_term_his, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent0 = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_ent1 = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)
        print(len(all_triples))

        long_history = long_term_his[0] + long_term_his[1]
        evolve_embs, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        # label = torch.zeros(len(all_triples), self.num_ents)
        # for idx, (h, r, t) in enumerate(all_triples):
        #     # print(h,r,t)
        #     # print(ans_dict[h.item()][r.item()])
        #     # if len(ans_dict[h.item()][r.item()]) == 0:
        #     #     print( "!!!!!!", h, r, t)
        #     # print(len(ans_dict[h.item()][r.item()]))
        #     # label[idx, list(ans_dict[h.item()][r.item()])] = 1
        #     label[idx, t] = 1
        # label = label.to(self.gpu)

        # print(len(all_triples), torch.sum(label))

        if self.method > 0:
            long_term_h = self.renet.forward(all_triples, 
                                        long_history, 
                                        self.emb_ent,
                                        self.emb_rel) 


        # if self.entity_prediction:
        # if  self.method == 0:
        #     scores_ent0 = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
        #     loss_ent0 = self.loss_e(scores_ent0, all_triples[:, 2])
        # elif self.method == 1:
        #     scores_ent0 = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
        #     scores_ent1 = self.linear_long(
        #             torch.cat((self.emb_ent_renet[all_triples[:, 0]], long_term_h,
        #                                 self.emb_rel_renet[all_triples[:, 1]]), dim=1))

        #     loss_ent0 = self.loss_e(scores_ent0, all_triples[:, 2])
        #     loss_ent1 = self.loss_e(scores_ent1, all_triples[:, 2])

        # elif self.method == 2:
        #     scores_ent0 = self.decoder_ob.forward(pre_emb, r_emb, long_term_h, all_triples)
        #     # scores_ent1 = self.linear_long(
        #     #         torch.cat((self.emb_ent_renet[all_triples[:, 0]], long_term_h,
        #     #                             self.emb_rel_renet[all_triples[:, 1]]), dim=1))
            
            
        #     loss_ent0 = self.loss_e(scores_ent0, all_triples[:, 2])
        #     # loss_ent1 = self.loss_e(scores_ent1, all_triples[:, 2])

        # elif self.method == 3:
        #     long_term_h = self.linear_hidden(
        #             torch.cat((self.emb_ent_renet[all_triples[:, 0]], long_term_h,
        #                                 self.emb_rel_renet[all_triples[:, 1]]), dim=1))
        #     score = self.decoder_ob.forward(pre_emb, r_emb, long_term_h, all_triples)
        #     loss_ent0 = self.loss_e(score, all_triples[:, 2])

        # elif self.method == 4:
        #     # scores_ent1 = self.linear_long(
        #     #         torch.cat((self.emb_ent_renet[all_triples[:, 0]], long_term_h,
        #     #                             self.emb_rel_renet[all_triples[:, 1]]), dim=1))
        #     scores_ent0 = self.decoder_ob.forward_v1(pre_emb, r_emb, long_term_h, all_triples, self.emb_ent_renet, self.emb_rel_renet)
        #     loss_ent0 = self.loss_e(scores_ent0, all_triples[:, 2])
        # elif self.method == 5:
        #     scores_ent0 = self.decoder_ob.forward_v2(pre_emb, r_emb, long_term_h, all_triples)
        #     loss_ent0 = self.loss_e(scores_ent0, all_triples[:, 2]) 

        # elif self.method == 6:
        #     scores_ent0 = self.decoder_ob.forward(pre_emb, r_emb, long_term_h, all_triples)
        #     loss_ent0 = self.loss_e(scores_ent0, all_triples[:, 2])

        # elif self.method == 7:
        #     scores_ent0 = self.decoder_ob.forward(pre_emb, r_emb, long_term_h, all_triples)
        #     loss_ent0 = self.loss_e(scores_ent0, all_triples[:, 2]) 

        # elif self.method == 8:
        scores_ent0 = self.decoder_ob.forward(pre_emb, r_emb, long_term_h, all_triples)
        loss_ent0 = self.loss_e(scores_ent0, all_triples[:, 2])  

             
        return loss_ent0, loss_ent1
