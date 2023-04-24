from cmath import phase
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from src.multievolve import MultiEvolve
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.renet import RENet
from src.decoder import ConvEvolve
import pickle


class MultiEvolveCross(MultiEvolve):
    def __init__(self, time_gap, time_start, seq_model, decoder_name, num_ents, num_rels, h_dim, t_dim, sequence_len, dropout=0, input_dropout=0, hidden_dropout=0, feat_dropout=0,
                 gpu=0, graph_dict=None):
        super(MultiEvolveCross, self).__init__(time_gap, time_start, seq_model, decoder_name, num_ents, num_rels, h_dim, t_dim, sequence_len, dropout, input_dropout, hidden_dropout, feat_dropout,
                 gpu, graph_dict)
        
        # attention network
        self.w_k = nn.Linear(h_dim, h_dim)
        self.w_v= nn.Linear(h_dim, h_dim)
        self.w_q = nn.Linear(2*h_dim, h_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # RENet for local history
        self.renet = RENet( self.num_ents, 
                            self.num_rels, 
                            self.time_gap,
                            self.time_start,
                            self.h_dim, 
                            self.t_dim,
                            self.hidden_dropout, 
                            seq_len=self.sequence_len, 
                            phase=2, 
                            seq_model=seq_model,
                            gpu=self.gpu)

    def attention_layer(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        mask = mask.unsqueeze(1).expand(-1, scores.size(1), -1)
        # print(scores.size(), mask.size())
        # print(mask==True)
        # print(scores)
        if mask is not None:
            scores = scores.masked_fill(mask == True, -1e9)
        
       
        # scores_zeros = (scores == 0)
        # # print(scores_zeros)
        # scores = scores.masked_fill(scores_zeros, -1e9)
        # print(scores)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn            
         
    def predict(self, time, test_triplets, ori_score, head_id, cands_snap, ori_ans_query, ans_query, long_term_his, cands_history):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels  # 将逆关系换成逆关系的id
            inv_relations = torch.cat((inverse_test_triplets[:, 1], test_triplets[:, 1]))
            test_triplets = torch.cat((test_triplets, inverse_test_triplets))
            scores = torch.sigmoid(ori_score.cpu())
            # long_history = [_[-self.sequence_len:] for _ in long_term_his[0][0]] + [_[-self.sequence_len:] for _ in long_term_his[1][0]]
            # long_history_t = [_[-self.sequence_len:] for _ in long_term_his[0][1]] + [_[-self.sequence_len:] for _ in long_term_his[1][1]]
            # hist = [cands_history[0][_][-self.sequence_len:] for _ in cands_snap]
            # hist_t = [cands_history[1][_][-self.sequence_len:] for _ in cands_snap]
            # long_history = long_term_his[0][0] + long_term_his[1][0]
            # long_history_t = long_term_his[0][1] + long_term_his[1][1]
            # print(len(cands_history[0]), len(cands_snap), torch.max(cands_snap))
            # hist = [cands_history[0][_] for _ in cands_snap]
            # hist_t = [cands_history[1][_] for _ in cands_snap]
            hist = [_[-self.sequence_len:] for _ in cands_history[0]]
            hist_t = [_[-self.sequence_len:] for _ in cands_history[1]] 

            query_h = self.renet.forward(time, test_triplets, long_term_his , self.emb_ent, self.emb_rel, None)    #[B, 2*h_dim]

            # query_h_att = torch.cat((query_h, self.emb_rel[test_triplets[:, 1]]), dim=1)
            query_h_att = torch.cat((self.emb_ent[test_triplets[:, 0]], self.emb_rel[inv_relations]), dim=1)
            # print(query_h_att.size())
            query_h_att = self.w_q(query_h_att)
            # query_h_att = self.emb_rel[inv_relations] 
            # query_h_att = self.emb_ent[test_triplets[:, 0]] + self.emb_rel[test_triplets[:, 1]] 
            # query_h_att = self.emb_rel[test_triplets[:, 1]] 
            evolve_emb_1, evolve_emb_2, pad_mask = self.renet.entity_forward(time, cands_snap, (hist, hist_t), self.graph_dict, self.emb_ent, self.emb_rel, None, phase=2)
            key = self.w_k(evolve_emb_2)
            value = self.w_v(evolve_emb_2)
            # key = evolve_emb_2
            # value = evolve_emb_2

            scores_list = []
            eval_bz = 1000
            num_triples = len(test_triplets)
            n_batch = (num_triples + eval_bz - 1) // eval_bz
            for idx in range(n_batch):
                batch_start = idx * eval_bz
                batch_end = min(num_triples, (idx + 1) * eval_bz)
                triples_batch = test_triplets[batch_start:batch_end, :]
                evolve_emb, atts = self.attention_layer(query_h_att[batch_start: batch_end, :], key, value, mask=pad_mask, dropout=None) # [Cand, B, H]
                # score_batch = score[batch_start:batch_end, :]
                # if rel_predict==1:
                #     target = test_triples[batch_start:batch_end, 1]
                # elif rel_predict == 2:
                #     target = test_triples[batch_start:batch_end, 0]
                # else:
                #     target = test_triples[batch_start:batch_end, 2]
                # rank.append(sort_and_rank(score_batch, target))
                # print(atts.size())
                # print(atts[:2,:1,:])
                # evolve_emb =  evolve_emb_1.unsqueeze(1).expand(evolve_emb_1.size(0), len(head_id), self.h_dim) + evolve_emb
                    
                # query_id = torch.LongTensor([_-batch_start for _ in range(batch_start, batch_end)]) 

                emb_list = [evolve_emb[ans_query[_], _-batch_start, :].unsqueeze(0) for _ in range(batch_start, batch_end)]
                ans_evolve_embs = torch.cat(emb_list, dim=0) # [B, topK ,H]
                ans_evolve_embs = ans_evolve_embs.transpose(0, 1)
                head_h = evolve_emb_1[head_id[batch_start: batch_end], : ]  # [B, H]

                evolve_emb_batch = [evolve_emb_1[ans_query[_], :].unsqueeze(0) for _ in range(batch_start, batch_end)]
                evolve_emb_batch = torch.cat(evolve_emb_batch, dim=0)
                evolve_emb_batch = evolve_emb_batch.transpose(0, 1)

                # ans_evolve_embs = evolve_emb_batch + ans_evolve_embs

                scores_part = self.decoder_ob.forward(self.emb_ent, self.emb_rel, head_h, query_h[batch_start: batch_end, :], ans_evolve_embs, triples_batch, phase=2) 
                scores_list.append(scores_part)
            scores_part = torch.cat(scores_list, dim=0)
            for _ in range(len(scores)):
                scores[_, ori_ans_query[_]] = 10000  + torch.sigmoid(scores_part[_, :].cpu())# + scores[idx, ori_ans_query[idx]] 
            return test_triplets.cpu(), scores.cpu()


    def get_loss(self, time, all_triples, head_id, cands_snap, long_term_his, cands_history, label):
        loss_ent0 = torch.zeros(1).cuda().to(self.gpu)
        # long_history = [_[-self.sequence_len:] for _ in long_term_his[0][0]] + [_[-self.sequence_len:] for _ in long_term_his[1][0]]
        # long_history_t = [_[-self.sequence_len:] for _ in long_term_his[0][1]] + [_[-self.sequence_len:] for _ in long_term_his[1][1]]
        # hist = [_[-self.sequence_len:] for _ in cands_history[0]]
        # hist_t = [_[-self.sequence_len:] for _ in cands_history[1]]

        query_h = self.renet.forward(time,
                        all_triples, 
                        long_term_his,
                        self.emb_ent,
                        self.emb_rel,
                        None, 
                        phase=2) 
        # print(query_h.size())

        batchsize = int(len(all_triples)/2)
        inv_relations = all_triples[:, 1].clone()
        inv_relations[:batchsize] = inv_relations[:batchsize] + self.num_rels
        inv_relations[batchsize:] = inv_relations[batchsize:] - self.num_rels 
        # print(inv_relations, all_triples[:, 1])
        
        query_h_att = torch.cat((self.emb_ent[all_triples[:, 0]], self.emb_rel[inv_relations]), dim=1)
        # query_h_att = torch.cat((query_h, self.emb_rel[all_triples[:, 1]]), dim=1)
        # print(query_h_att.size())
        query_h_att = self.w_q(query_h_att)
        # query_h_att = self.emb_rel[inv_relations] 
        # query_h_att = self.emb_ent[all_triples[:, 0]] + self.emb_rel[all_triples[:, 1]] 
        # query_h_att = self.emb_rel[all_triples[:, 1]] 
        
        evolve_emb_1, evolve_emb_2, pad_mask = self.renet.entity_forward(time, cands_snap, cands_history, self.graph_dict, self.emb_ent, self.emb_rel, None, phase=2)
        
        # print(evolve_emb_1.size(), evolve_emb_2.size())
        key = self.w_k(evolve_emb_2)
        value = self.w_v(evolve_emb_2)
        # key = evolve_emb_2
        # value = evolve_emb_2

        # print(evolve_emb_2.size())
        evolve_emb, atts = self.attention_layer(query_h_att, key, value, mask=pad_mask, dropout=self.dropout_layer)
        # print(evolve_emb.size())
        # evolve_emb [E, N, h]
        # evolve_emb =  evolve_emb_1.unsqueeze(1).expand(evolve_emb_1.size(0), len(head_id), self.h_dim) + evolve_emb
                
        # query_id = [_ for _ in range(len(head_id))]
        # head_h = evolve_emb[head_id, query_id, :]  # [N, h]
        head_h = evolve_emb_1[head_id]

        # evolve_emb =  evolve_emb_1.unsqueeze(1).expand(evolve_emb_1.size(0), len(head_id), self.h_dim) + evolve_emb
                

        scores = self.decoder_ob.forward(self.emb_ent, self.emb_rel, head_h, query_h, evolve_emb, all_triples, phase=2)

        loss_ent0 = self.loss_e(scores, label)

        return loss_ent0
