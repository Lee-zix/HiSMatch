import torch.nn as nn
import numpy as np
import torch
from aggregator import MeanAggregator, AttnAggregator, RGCNAggregator


class EvolveNet(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim, renet_dropout_rate, seq_len,
                 method, gpu):
        super(EvolveNet, self).__init__()
        # assert (args.entity_dim == args.relation_dim)
        self.h_dim = hidden_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.renet_dropout_rate = renet_dropout_rate
        self.seq_len = seq_len
        self.method = method
        self.gpu = gpu

        self.dropout = nn.Dropout(self.renet_dropout_rate)
        self.sub_encoder = nn.GRU(3 * self.h_dim,  self.h_dim, batch_first=True)
        self.node_encoder = nn.GRU(3 * self.h_dim,  self.h_dim, batch_first=True)
        # self.ob_encoder = self.sub_encoder

        if self.method == 0: # Attentive Aggregator
            self.aggregator_s = AttnAggregator(self.h_dim, self.renet_dropout_rate, self.seq_len)
        elif self.method == 1: # Mean Aggregator
            self.aggregator_s = MeanAggregator(self.h_dim, self.renet_dropout_rate, self.seq_len, gcn=False)

        # rgcn aggregator 
        self.aggregator_rgcn = RGCNAggregator(self.h_dim, self.renet_dropout_rate,
                                              self.num_entities, self.num_relations, 50,
                                              seq_len=10, gpu=self.gpu)

        self.linear_sub = nn.Linear(3 * self.h_dim, self.num_entities)
        # self.linear_ob = self.linear_sub

        self.criterion = nn.CrossEntropyLoss()

    """
    Prediction function in training. 
    This should be different from testing because in testing we don't use ground-truth history.
    def forward(self, all_triples, s_hist, entity_embeddings, relation_embeddings):
        '''
        :param triplets:list with each element :[head, tail, rel, time]
        :param s_hist:  history of each element : list of list [[tail_t1_0, tail_t1_1],[tail_t2_1],[tail_t3_1, tail_t3_2],[]..,[]]
        :param kg:  object of knowledge graph class kg.graph_dict = None if model = 1,2
        :return:
        '''
        # print(all_triples.size())
        s, r, o = all_triples[:, 0], all_triples[:, 1], all_triples[:, 2]
        s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
        # print(s_hist_len)
        s_len, s_idx = s_hist_len.sort(0, descending=True)
        if s_len[0] > 0:
            s_packed_input = self.aggregator_s.forward(s_hist, s, r, entity_embeddings,
                                               relation_embeddings)
            # print(s_packed_input.size())
            _, s_h = self.sub_encoder(s_packed_input)
            s_h = s_h.squeeze(0)
            s_h = torch.cat((s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()), dim=0)
        else:
            s_h = torch.zeros(len(s) , self.h_dim).cuda()
        # print(s_h.size())
        # print(entity_embeddings[s[s_idx]].size())
        ob_pred = self.linear_sub(
            self.dropout(torch.cat((entity_embeddings[s[s_idx]], s_h,
                                    relation_embeddings[r[s_idx]]), dim=1)))
        inverse_s_idx = torch.zeros(len(s_idx)).long()
        for i, s in enumerate(s_idx.tolist()):
            inverse_s_idx[s] = i
        ob_pred_inverse = ob_pred[inverse_s_idx]
        # return ob_pred, s_idx
        # loss_ob = self.criterion(ob_pred, o)
        # loss = loss_sub + loss_ob
        return ob_pred_inverse
    """
    
    def entity_forward(self, t_idx, all_triples, short_term_history, graph_dict, entity_embeddings, relation_embeddings):
        # uniq_v, edges = torch.unique(all_triples[:,[0, 2]], return_inverse=True)
        s, r = all_triples[:, 0], all_triples[:, 1]
        
        # generate history for each unique node
        short_histories = []
        short_histories_t = []
        for v in s:
            short_histories.append(short_term_history[(t_idx, v.item())][0])
            short_histories_t.append(short_term_history[(t_idx, v.item())][1])
        
        # get the history for all (s, r) pairs (including ducplicate triples)
        s_hist_len = torch.LongTensor(list(map(len, short_histories))).cuda()
        s_len, s_idx = s_hist_len.sort(0, descending=True)
        if s_len[0] > 0:
            s_packed_input = self.aggregator_rgcn.forward((short_histories, short_histories_t),
                                                          s,
                                                          r, 
                                                          entity_embeddings,
                                                          relation_embeddings, 
                                                          graph_dict)
            # print(s_packed_input.data.size())
            _, s_h = self.node_encoder(s_packed_input)
            s_h = s_h.squeeze(0)
            s_h = torch.cat((s_h, torch.zeros(len(s) - len(s_h),  2 * self.h_dim).cuda()), \
                  dim=0)
        else:
            s_h = torch.zeros(len(s), 2 * self.h_dim).cuda()

    
        # restitute the node order to the original order
        inverse_s_idx = torch.zeros(len(s_idx)).long()
        for i, s in enumerate(s_idx.tolist()):
            inverse_s_idx[s] = i
        s_h = s_h[inverse_s_idx]

        # get all triples' [s, r] embeddings
        # print(s.size(), s_h.size())

        # short_entity = s_h[all_triples[:, 0]]
        # short_relation = relation_embeddings[all_triples[:, 1]]
        # print(short_entity.size(), short_relation.size())
        # e1_embedded = short_entity.unsqueeze(1)
        # rel_embedded = short_relation.unsqueeze(1)
        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        return s_h

    def forward(self, all_triples, s_hist, entity_embeddings, relation_embeddings):
        '''
        :param triplets:list with each element :[head, tail, rel, time]
        :param s_hist:  history of each element : list of list [[tail_t1_0, tail_t1_1],[tail_t2_1],[tail_t3_1, tail_t3_2],[]..,[]]
        :param kg:  object of knowledge graph class kg.graph_dict = None if model = 1,2
        :return:
        '''
        # print(all_triples.size())
        s, r, o = all_triples[:, 0], all_triples[:, 1], all_triples[:, 2]
        s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
        # print(s_hist_len)
        s_len, s_idx = s_hist_len.sort(0, descending=True)
        if s_len[0] > 0:
            s_packed_input = self.aggregator_s.forward(s_hist, s, r, entity_embeddings,
                                                       relation_embeddings)
            # print(s_packed_input.size())
            _, s_h = self.sub_encoder(s_packed_input)
            s_h = s_h.squeeze(0)
            s_h = torch.cat((s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()), dim=0)
        else:
            s_h = torch.zeros(len(s) , self.h_dim).cuda()

        # ob_pred = self.linear_sub(
        #     self.dropout(torch.cat((entity_embeddings[s[s_idx]], s_h,
        #                             relation_embeddings[r[s_idx]]), dim=1)))
        inverse_s_idx = torch.zeros(len(s_idx)).long()
        for i, s in enumerate(s_idx.tolist()):
            inverse_s_idx[s] = i
        # ob_pred_inverse = ob_pred[inverse_s_idx]
        local_hidden =  s_h[inverse_s_idx]
        return local_hidden


    def predict_fact(self, s, r, o, t, pred_e2, s_hist, kg):
        """
        forward a batch of examples
        :param examples: [h, r, t]
        :param history: triples in the history
        :return:
        """
        t = t.cpu()
        if len(s_hist[0]) == 0:
            s_h = torch.zeros(self.h_dim).cuda()
        else:
            s_history = s_hist[0]
            s_history_t = s_hist[1]
            # print(s_history)
            # print(s_history_t)
            inp = self.aggregator_s.predict((s_history, s_history_t), s, r, t,
                                            kg.get_all_entity_embeddings(),
                                            kg.get_all_relation_embeddings(), self.graph_dict,
                                            reverse=False)
            tt, s_h = self.sub_encoder(inp.view(1, len(s_history), 3 * self.h_dim))
            s_h = s_h.squeeze()

        ob_pred = self.linear_sub(torch.cat((self.kg.get_entity_embeddings(s), s_h,
                                             kg.get_relation_embeddings(r)),
                                            dim=0))
        return ob_pred

def get_renet_nn_state_dict(state_dict):
    renet_nn_state_dict = {}
    for param_name in ['mdl.sub_encoder.weight_ih_l0',
                       'mdl.sub_encoder.weight_hh_l0',
                       'mdl.sub_encoder.bias_ih_l0',
                       'mdl.sub_encoder.bias_hh_l0',
                       'mdl.aggregator_s.rgcn1.loop_weight',
                       'mdl.aggregator_s.rgcn1.weight',
                       'mdl.aggregator_s.rgcn2.loop_weight',
                       'mdl.aggregator_s.rgcn2.weight',
                       'mdl.linear_sub.weight',
                       'mdl.linear_sub.bias']:
        renet_nn_state_dict[param_name.split('.', 1)[1]] = state_dict['state_dict'][param_name]
    return renet_nn_state_dict

def get_renet_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        kg_state_dict[param_name.split('.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict


