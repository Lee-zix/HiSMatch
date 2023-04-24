import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, UnionRGATLayer
from src.model import BaseRGCN
from src.renet import RENet
from src.decoder import ConvEvolve
import pickle
from rgcn.utils import get_one_month_day_hour_min
from torch.distributions import Categorical, kl


class MultiEvolve(nn.Module):
    def __init__(self, time_gap, time_start, seq_model, encoder, num_ents, num_rels, h_dim, t_dim, sequence_len,
                 dropout=0, input_dropout=0, hidden_dropout=0, feat_dropout=0, bg_encoder="uvrgcn",
                 gpu=0, graph_dict=None):
        super(MultiEvolve, self).__init__()
        self.encoder = encoder
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.t_dim = t_dim
        self.hidden_dropout = hidden_dropout
        self.h = None
        self.relation_evolve = False
        self.emb_rel = None
        self.gpu = gpu
        self.graph_dict = graph_dict
        self.time_gap = time_gap
        self.time_start = time_start
        self.bg_encoder = bg_encoder

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.emb_ent = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.emb_ent)

        if self.bg_encoder == "rgcn":
            self.bg_rgcn_layer_1 = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_rels*2, num_bases=32,
                                    activation=F.rrelu, dropout=dropout, self_loop=True, skip_connect=False)
            self.bg_rgcn_layer_2 = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_rels*2, num_bases=32,
                            activation=F.rrelu, dropout=dropout, self_loop=True, skip_connect=False)
        elif self.bg_encoder.startswith("uvrgcn"):
            comp = self.bg_encoder.split("-")[1]
            print("!!!", comp)
            self.bg_rgcn_layer_1 = UnionRGCNLayer(self.h_dim, self.h_dim, 2*self.num_rels, comp, num_bases=32,
                            activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)
            self.bg_rgcn_layer_2 = UnionRGCNLayer(self.h_dim, self.h_dim, 2*self.num_rels, comp, num_bases=32,
                               activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)
        elif self.bg_encoder == "kbat":
            self.bg_rgcn_layer_1 = UnionRGATLayer(self.h_dim, self.h_dim, 2*self.num_rels, num_bases=32,
                            activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)
            self.bg_rgcn_layer_2 = UnionRGATLayer(self.h_dim, self.h_dim, 2*self.num_rels, num_bases=32,
                               activation=F.rrelu, self_loop=True, dropout=dropout, skip_connect=False)

        # RENet for local history
        self.renet = RENet( self.num_ents, 
                            self.num_rels, 
                            self.time_gap,
                            self.time_start,
                            self.h_dim, 
                            self.t_dim,
                            self.hidden_dropout, 
                            seq_len=self.sequence_len, 
                            phase=1, 
                            encoder_name=self.encoder,
                            seq_model=seq_model,
                            gpu=self.gpu)

        self.loss_e = torch.nn.CrossEntropyLoss()
        self.w_global = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.h_dim))).float())
        self.b_global = torch.nn.Parameter(torch.zeros(self.h_dim).float()) 

        self.fcn_global = torch.nn.Linear(self.h_dim, self.num_ents)
        self.decoder_ob = ConvEvolve(num_ents, h_dim, t_dim, input_dropout, hidden_dropout, feat_dropout)

        
    def predict(self, time, global_g, test_triplets, head_id, cands_snap, packed_query_input, packed_node_input):
        with torch.no_grad():
            # month, day, week = get_one_month_day_hour_min(time, gap=1, start=1)
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels  # 将逆关系换成逆关系的id
            test_triplets = torch.cat((test_triplets, inverse_test_triplets))
            
            if global_g is None:
                emb_ent = self.emb_ent
            else:
                global_g = global_g.to(self.gpu)
                global_g.ndata['h'] = self.emb_ent  # 演化得到的表示，和wordemb满足静态图约束

                if self.bg_encoder == "rgcn":
                    self.bg_rgcn_layer_1(global_g, self.emb_ent)
                    self.bg_rgcn_layer_2(global_g, global_g.ndata['h'])
                elif self.bg_encoder.startswith("uvrgcn") or self.bg_encoder == "kbat":
                    self.bg_rgcn_layer_1(global_g, self.emb_ent, self.emb_rel.data)
                    self.bg_rgcn_layer_2(global_g, global_g.ndata['h'], self.emb_rel.data)
                emb_ent = global_g.ndata.pop('h')  

            query_h = self.renet.forward(time, test_triplets, packed_query_input, emb_ent, self.emb_rel, None)    #[B, 2*h_dim]
            evolve_emb = self.renet.entity_forward(time, cands_snap, packed_node_input, self.graph_dict, emb_ent, self.emb_rel, None)
            # evolve_emb = evolve_emb + emb_ent
            head_h = evolve_emb[head_id]

            # head_h = emb_ent[head_id]
            # hidden = torch.cat([query_h, head_h], dim=1)
            # scores = self.fcn_global(query_h)

            # global_t = (time - self.time_start)/self.time_gap
            # global_t = torch.sin(time * self.w_global + self.b_global)
            # hidden = global_t.unsqueeze(0).repeat(len(test_triplets), 1)
            # # print(all_triples)
            # hidden = torch.cat([self.emb_ent_g[test_triplets[:, 0]], self.emb_rel_g[test_triplets[:, 1]], hidden], dim=1)
            # global_scores = self.fcn_global(hidden)

            # day_h = self.emb_day[week].unsqueeze(0).repeat(len(test_triplets), 1)
            scores = self.decoder_ob.forward(emb_ent, self.emb_rel, head_h, query_h, evolve_emb, test_triplets) 

            # scores = torch.softmax(scores, dim=1)
            # global_scores = torch.softmax(global_scores, dim=1)

            # scores = 0.8*scores + 0.2*global_scores
        
            return test_triplets.cpu(), scores.cpu()

    def get_loss(self, time, global_g, triples, head_id, cands_snap, packed_query_input, packed_node_input, label):
        # loss_ent0 = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        # loss_ent1 = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        # month, day, week = get_one_month_day_hour_min(time, gap=1, start=1)
        # print(month, day, week)
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        if global_g is None:
            emb_ent = self.emb_ent
        else:
            global_g = global_g.to(self.gpu)
            global_g.ndata['h'] = self.emb_ent  # 演化得到的表示，和wordemb满足静态图约束
            if self.bg_encoder == "rgcn":
                self.bg_rgcn_layer_1(global_g, self.emb_ent)
                self.bg_rgcn_layer_2(global_g, global_g.ndata['h'])
            elif self.bg_encoder.startswith("uvrgcn") or self.bg_encoder == "kbat":
                self.bg_rgcn_layer_1(global_g, self.emb_ent, self.emb_rel.data)
                self.bg_rgcn_layer_2(global_g, global_g.ndata['h'], self.emb_rel.data)
                
            emb_ent = global_g.ndata.pop('h') 
        # emb_ent = F.normalize(emb_ent)
        # emb_ent = self.emb_ent

        # global_t = (time - self.time_start)/self.time_gap
        # global_t = torch.sin(time * self.w_global + self.b_global)
        # hidden = global_t.unsqueeze(0).repeat(len(all_triples), 1)
        # print(all_triples)
        # hidden = torch.cat([self.emb_ent_g[all_triples[:, 0]], self.emb_rel_g[all_triples[:, 1]], hidden], dim=1)
        # global_scores = self.fcn_global(hidden) 

        # long_history = [_[-self.sequence_len:] for _ in long_term_his[0][0]] + [_[-self.sequence_len:] for _ in long_term_his[1][0]]
        # long_history_t = [_[-self.sequence_len:] for _ in long_term_his[0][1]] + [_[-self.sequence_len:] for _ in long_term_his[1][1]]
        # hist = [cands_history[0][_][-self.sequence_len:] for _ in cands_snap]
        # hist_t = [cands_history[1][_][-self.sequence_len:] for _ in cands_snap]

        query_h = self.renet.forward(time, all_triples, packed_query_input, emb_ent, self.emb_rel, None) 
        evolve_emb = self.renet.entity_forward(time, cands_snap, packed_node_input, self.graph_dict, emb_ent, self.emb_rel, None)
        # evolve_emb = evolve_emb + emb_ent
        head_h = evolve_emb[head_id]
        # head_h = emb_ent[head_id]

        # hidden = torch.cat([query_h, head_h], dim=1)
        # scores = self.fcn_global(query_h)

        # day_h = self.emb_day[week].unsqueeze(0).repeat(len(all_triples), 1)
        # print(day_h.size())
        scores = self.decoder_ob.forward(emb_ent, self.emb_rel, head_h, query_h, evolve_emb, all_triples)
        # print(scores.size())
        # print(label)
        loss1 = self.loss_e(scores, label)
        # loss2 = self.loss_e(global_scores, label)

        # distri = torch.softmax(global_scores, dim=1)
        # loss3 =0 
        # for i in range(len(distri)):
        #     loss3 += sum([distri[i, _] * torch.log(distri[i, _]) for _ in range(len(distri[i]))])
        # print(loss1, loss2, loss3)
        return loss1
