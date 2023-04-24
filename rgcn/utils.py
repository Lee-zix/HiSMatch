"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch
import dgl
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import copy
import os
import pickle
import datetime
import sys

def pre_packed_query_input(snap_list, s_hist, o_hist, start_time, time_gap, num_rels, history_len):
    idx = [_ for _ in range(len(snap_list))]
    packed_input_list = []
    for train_sample_num in tqdm(idx):
        ab_time = start_time+train_sample_num*time_gap
        output = snap_list[train_sample_num]

        his_begin = sum([len(_) for _ in snap_list[0: train_sample_num]])
        his_end = his_begin + len(snap_list[train_sample_num])
        s_history = [s_hist[0][_] for _ in range(his_begin, his_end)]
        s_history_t = [s_hist[1][_] for _ in range(his_begin, his_end)]
        
        o_history = [o_hist[0][_] for _ in range(his_begin, his_end)]
        o_history_t = [o_hist[1][_] for _ in range(his_begin, his_end)]
        inverse_triples = output[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        all_triples = torch.cat([output, inverse_triples])
        long_history = [_[-history_len:] for _ in s_history] + [_[-history_len:] for _ in o_history]
        long_history_t = [_[-history_len:] for _ in s_history_t] + [_[-history_len:] for _ in o_history_t]
       
        # s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, neigh_num_of_s, flat_s, s_len_sorted, s_idx
        packed_output = get_sorted_s_r_embed((long_history, long_history_t), all_triples[:, 0], all_triples[:, 1], ab_time, history_len, time_gap, start_time)
        packed_input_list.append(packed_output) 
    return packed_input_list


def pre_packed_node_input(snap_list, node_list, cands_history, graph_dict, start_time, time_gap, num_rels, history_len):
    idx = [_ for _ in range(len(snap_list))]
    packed_input_list = []
    for train_sample_num in tqdm(idx):
        ab_time = start_time+train_sample_num*time_gap

        hist = [cands_history[0][train_sample_num][_][-history_len:] for _ in node_list]
        hist_t = [cands_history[1][train_sample_num][_][-history_len:] for _ in node_list]

        # g_list, node_ids_graph, s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, padding_mask, s_idx
        packed_output = get_sorted_s_r_embed_rgcn((hist, hist_t), node_list, ab_time, graph_dict, history_len, time_gap, start_time)
        packed_input_list.append(packed_output) 

    return packed_input_list

def add_history(triplets_list, init_start_time, start_time, time_gap, num_rels, num_nodes, cand_ans, history_dict, history_t_dict):
    candidate_list = []
    history_list = []
    history_t_list = []
    for retime, triplets in enumerate(tqdm(triplets_list)):
        time = time_gap*retime + start_time
        inverse_triplets = triplets[:, [2, 1, 0]]
        inverse_triplets[:, 1] = inverse_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
        all_triplets = np.concatenate((triplets, inverse_triplets), axis=0)
        if cand_ans is not None:
            candidates = []
            for idx in range(len(all_triplets)):
                assert(len(all_triplets) == len(cand_ans[retime]))
                cand = cand_ans[retime][idx, :]
                # print(len(cand))
                # ent = np.concatenate((all_triplets[:, 0], all_triplets[:, 2]))
                candidates.extend(np.unique(cand).tolist())
            candidates.extend(all_triplets[:, 0])
            candidates.extend(all_triplets[:, 2])
            candidates = set(candidates) 
        else:
            candidates = [_ for _ in range(num_nodes)]
        candidate_history = []
        candidate_history_t = []
        for ent_ in candidates:
            # TODO time start from 0 or 1 or other numbers ????
            t = time
            while(t > init_start_time):
            # for t in range(time, -1, -time_gap):
                if (ent_, t) in history_dict:
                    candidate_history.append(history_dict[(ent_, t)].copy())
                    candidate_history_t.append(history_t_dict[(ent_, t)].copy())
                    break
                t = t - time_gap
            if t == init_start_time:
                candidate_history.append([])
                candidate_history_t.append([])
        candidate_list.append(list(candidates))
        history_list.append(candidate_history)
        history_t_list.append(candidate_history_t)
        assert(len(candidates)==len(candidate_history)==len(candidate_history_t))
    return candidate_list, history_list, history_t_list 

def label_transfer(query_ans, snap_list, snap_cand_list, num_rels, mode):
    """label_transfer: In 2-phase, translate the label id to ent id

    Args:
        query_ans ([type]): [description]
        snap_list ([type]): [description]
        snap_cand_list ([type]): [description]

    Returns:
        [head_list, snap_cand_list, ori_ans]: [description]
    """
    # ori_ans = copy.deepcopy(query_ans)
    ans =[]
    head_list = []
    label_list = []
    for cand, triples, cand_snap in zip(query_ans, snap_list, snap_cand_list):
        label_snap = [] 
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        all_triples = np.concatenate((triples, inverse_triples), axis=0)
        ent2id = {ent: idx for idx, ent in enumerate(cand_snap)}
        # id2ent = {idx: ent for idx, ent in enumerate(cand_snap)}
        # head entitiy
        head = [ent2id[_] for _ in all_triples[:, 0]]
        # print(cand[0])
        cand_ans = []
        for idx, (_, _, o) in enumerate(all_triples):
            cand_ans.append([ent2id[cand_] for cand_ in cand[idx]])
            if mode == "train":
                label_snap.append(np.where(cand_snap == o)[0][0]) 
        if mode == "train":
            label_list.append(label_snap) 
        ans.append(torch.Tensor(cand_ans).long())
        head_list.append(head)
    # head_list = [torch.Tensor(_).long() for _ in head_list]
    # ori_ans = [torch.Tensor(_).long() for _ in ori_ans]
    # snap_cand_list = [torch.Tensor(_).long() for _ in snap_cand_list]
    # label_list = [torch.Tensor(_).long().cuda() for _ in label_list]
    return head_list, snap_cand_list, ans, label_list


def load_his_data(history_data_dir, test=False):
    with open(os.path.join(history_data_dir, 'dev_history_sub.txt'), 'rb') as f:
        s_history_dev_data = pickle.load(f)
    with open(os.path.join(history_data_dir, 'dev_history_ob.txt'), 'rb') as f:
        o_history_dev_data = pickle.load(f)
    if test:
        s_history_train_data, o_history_train_data = None, None
        with open(os.path.join(history_data_dir, 'test_history_sub.txt'), 'rb') as f:
            s_history_test_data = pickle.load(f)
        with open(os.path.join(history_data_dir, 'test_history_ob.txt'), 'rb') as f:
            o_history_test_data = pickle.load(f)
    else:
        s_history_test_data, o_history_test_data = None, None  
        with open(os.path.join(history_data_dir, 'train_history_sub.txt'), 'rb') as f:
            s_history_train_data = pickle.load(f)
        with open(os.path.join(history_data_dir, 'train_history_ob.txt'), 'rb') as f:
            o_history_train_data = pickle.load(f)
   
    return s_history_train_data, o_history_train_data, s_history_dev_data, o_history_dev_data, s_history_test_data, o_history_test_data 
    
def load_ans_for_eval(data, num_rels, num_nodes, ans_type):
    """Entity prediction: ans type = False. Relation prediction: ans_type = True """
    if ans_type == False:
        all_ans_list_train = load_all_answers_for_time_filter(data.train, num_rels, num_nodes, False)
        all_ans_list_valid = load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
        all_ans_list_test = load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    else:
        all_ans_list_train = load_all_answers_for_time_filter(data.train, num_rels, num_nodes, True)
        all_ans_list_valid = load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
        all_ans_list_test = load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True) 
    return all_ans_list_train, all_ans_list_valid, all_ans_list_test


def move_dgl_to_cuda(g):
    # g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
    # g.edata.update({k: cuda(g.edata[k]) for k in g.edata})
    g.to("cuda:"+str(torch.cuda.current_device()))


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################


def get_month_day_hour_min(time_list, gap, start, begin_time="2014-01-01", dataset="ICEWS"):
    month_list, day_list, week_list = [], [], []
    if dataset.startswith("ICEWS"):
        for delta in time_list:
            startTime = datetime.datetime.strptime(begin_time, "%Y-%m-%d")
            now_time = startTime + datetime.timedelta(days=int((delta-start)/gap))
            month_list.append(int(now_time.month))
            day_list.append(int(now_time.day))
            week_list.append(int(now_time.weekday()))
    else:
        print("TO DO")
    return month_list, day_list, week_list

def get_one_month_day_hour_min(time, gap, start, begin_time="2014-01-01", dataset="ICEWS"):
    if dataset.startswith("ICEWS"):
        startTime = datetime.datetime.strptime(begin_time, "%Y-%m-%d")
        now_time = startTime + datetime.timedelta(days=int((time-start)/gap))
        month = int(now_time.month)
        day = int(now_time.day)
        week = int(now_time.weekday())
    else:
        print("TO DO")
    return month, day, week

def get_sorted_s_r_embed(hist, s, r, t, seq_len, time_gap, time_start):
    s_hist = hist[0]
    s_hist_t = hist[1]
    
    s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
    s_len_sorted, s_idx = s_hist_len.sort(0, descending=True)
    #  torch.nonzeros: Returns a tensor containing the indices of all non-zero elements of input
    num_non_zero = len(torch.nonzero(s_len_sorted))
    #   Return all non-zero elements of input
    s_len_non_zero = s_len_sorted[:num_non_zero]
    # s_len_non_zero: all non-zeros element of input
    s_hist_sorted = []
    s_hist_t_sorted = []
    s_time_iterval_sorted = []
    s_time_month = []
    s_time_day = []
    s_time_week = []
    for i, idx in enumerate(s_idx):
        # idx.item :  tensor convert to python scalars
        if i == num_non_zero:
            break
        s_hist_sorted.append(s_hist[idx.item()])
        s_hist_t_sorted.append(s_hist_t[idx.item()])
        target_time = [t] * len(s_hist_t[idx])
        target_time = torch.Tensor(target_time).cuda()
        hist_tensor = torch.Tensor(s_hist_t[idx]).cuda()
        # print("target", target_time)
        # print("hist", hist_tensor)
        s_time_iterval_sorted.append((target_time - hist_tensor)/time_gap)
        month_sorted, day_sorted, week_sorted = get_month_day_hour_min(s_hist_t[idx], time_gap, time_start)
        s_time_month.append(torch.Tensor(month_sorted))
        s_time_day.append(torch.Tensor(day_sorted))
        s_time_week.append(torch.Tensor(week_sorted))
    flat_s = []
    neigh_num_of_s = []
    s_hist_sorted = s_hist_sorted[:num_non_zero]
    for hist in s_hist_sorted:  # each example
        for neighs in hist:     # each hist of the example
            neigh_num_of_s.append(len(neighs))   # len_s [3, 2, 1, 1, 2]
            # print(neighs)
            # for neigh in neighs:    # each e in hist
                # flat_s [[[1, 2, 3], [4, 5], [1]], [[4], [8, 9]]]-->[1, 2, 3, 4, 5, 1, 4, 8, 9]
                # print(neigh)
            flat_s.extend(neighs[:, -1])
    # sort samples in batch according to the length of history
    # s_tem = s[s_idx]
    # r_tem = r[s_idx]
    padding_mask = torch.ones(num_non_zero, seq_len+1).cuda()
    for i in range(num_non_zero):
        padding_mask[i, :s_len_sorted[i]+1] = 0
    return s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, s_time_week, padding_mask, neigh_num_of_s, flat_s, s_len_sorted, s_idx

def get_neighs_by_t(s_hist_sorted, s_hist_t_sorted):
    """
    len_s: list of number entities in all subgraphs
    max_len: max entity number
    node_list: list of list 
    """   
    neighs_t = defaultdict(set)
    len_s = []
    node_list = []
    for i, (hist, hist_t) in enumerate(zip(s_hist_sorted, s_hist_t_sorted)):
        node_list__ = []
        for neighs, t in zip(hist, hist_t):
            # for nei in neighs[:, 2]:
            #     if nei not in graph_dict[t].ids.keys():
            #         print("tail not in graph", s_tem[i].item(), nei, t)
                # if s_tem[i].item() not in graph_dict[t].ids.keys():
                #     print("s_tem ent not in graph ", i, s_tem[i].item(), r_tem[i].item(),
                #           t_tem[i].item(), t)
                #     print("hist", hist)
                #     print("hist_t", hist_t)
            # for nei in neighs[:, 0]:
            #     if nei not in graph_dict[t].ids.keys():
            #         print("head not in graph", s_tem[i].item(), nei, t)
                # if s_tem[i].item() not in graph_dict[t].ids.keys():
                #     print("s_tem ent not in graph ", i, s_tem[i].item(), r_tem[i].item(),
                #           t_tem[i].item(), t)
                #     print("hist", hist)
                #     print("hist_t", hist_t)
            head = neighs[:, 0].tolist()
            tail = neighs[:, 2].tolist()
            neighs_t[t].update(head)
            neighs_t[t].update(tail)
            ents = list(set(head+tail))
            len_s.append(len(ents))
            node_list__.append(ents)
            # neighs_t[t].add(s_tem[i].item())
        node_list.append(node_list__)
    max_len = max(len_s)
    len_s = torch.tensor(len_s).view(-1, 1).float()
    return neighs_t, len_s, node_list, max_len

def get_g_list_id(neighs_t, graph_dict):
    """AI is creating summary for get_g_list_id

    Args:
        neighs_t ([type]): [description]
        graph_dict ([type]): [description]

    Returns:
        glist [list]: [generate a dgl.subgraph for each timestamp in a batch]
        g_id_dict [dict]: [dict from timestamps to idx in g_list]
        nodes_num: [int]: [all nodes number in a batch (note that the nodes includes duclipate entities)] 
    """
    g_id_dict = {}
    g_list = []
    idx = 0
    nodes_num = 0
    # print("--------check for key error-----------")
    for tim in neighs_t.keys():
        g_id_dict[tim] = idx
        # print(tim)
        # print(neighs_t[tim])
        # print(graph_dict[tim].ids)
        # for _ in neighs_t[tim]:
        #     if _ not in graph_dict[tim].ids:
        #         print('wrong', tim, _, graph_dict[tim])
        g_list.append(make_subgraph(graph_dict[tim], neighs_t[tim]))
        # print(g_list[idx].ids)
        if idx == 0:
            g_list[idx].start_id = 0
        else:
            g_list[idx].start_id = g_list[idx - 1].start_id + g_list[idx - 1].number_of_nodes()
            nodes_num +=  g_list[idx-1].number_of_nodes()
        idx += 1
    nodes_num += g_list[-1].number_of_nodes()
    return g_list, g_id_dict, nodes_num


# def get_node_ids_to_g_id(s_hist_t_sorted, g_list, g_id_dict, node_list, max_len, all_nodes_num):
#     """AI is creating summary for get_node_ids_to_g_id

#     Args:
#         s_hist_t_sorted ([type]): [description]
#         g_list ([type]): [description]
#         g_id_dict ([type]): [description]
#         node_list ([type]): [description]
#         max_len ([type]): [description]
#         all_nodes_num ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     node_ids_graph = []
#     for i, nodes_ in enumerate(node_list):
#         for j, nodes in enumerate(nodes_):
#             node_ids = []
#             t = s_hist_t_sorted[i][j]
#             graph=g_list[g_id_dict[t]]
#             for node in nodes:
#                 node_ids.append(graph.ids[node] + graph.start_id)
#             node_ids_graph.append(torch.tensor(node_ids))
#     # node_ids_graph = torch.tensor(node_ids_graph)
#     node_ids_graph = pad_sequence(node_ids_graph, max_len, all_nodes_num)
#     node_ids_graph = node_ids_graph.view(-1)
#     return node_ids_graph


def get_node_ids_to_g_id(s_hist_sorted, s_hist_t_sorted, s_tem, g_list, g_id_dict, max_len, all_nodes_num):

    node_ids_graph = []
    len_s = []
    for i, hist in enumerate(s_hist_sorted):
        node_ids = []
        for j, neighs in enumerate(hist):
            len_s.append(len(neighs))
            t = s_hist_t_sorted[i][j]
            graph = g_list[g_id_dict[t]]
            node_ids.append(graph.ids[s_tem[i].item()] + graph.start_id)
        node_ids_graph.append(torch.LongTensor(node_ids))
    # print(all_nodes_num)
    node_ids_graph = pad_sequence(node_ids_graph, batch_first=True, padding_value=int(all_nodes_num))
    if node_ids_graph.size(1) < max_len:
        node_ids_graph = torch.cat([node_ids_graph, torch.zeros(node_ids_graph.size(0), max_len-node_ids_graph.size(1)).long()], dim=1)
    # node_ids_graph = node_ids_graph.view(-1)
    return node_ids_graph


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def make_subgraph(g, nodes):
    nodes = list(nodes)
    relabeled_nodes = []
    # print('g.ids', g.ids)
    for node in nodes:
        relabeled_nodes.append(g.ids[node])

    sub_g = g.subgraph(relabeled_nodes)

    #sub_g.ndata.update({k: g.ndata[k][sub_g.parent_nid] for k in g.ndata if k != 'norm'})
    sub_g.ndata.update({k: g.ndata[k][sub_g.ndata[dgl.NID]] for k in g.ndata if k != 'norm'})
    #sub_g.edata.update({k: g.edata[k][sub_g.parent_eid] for k in g.edata})
    sub_g.edata.update({k: g.edata[k][sub_g.edata[dgl.EID]] for k in g.edata})
    sub_g.ids = {}
    norm = comp_deg_norm(sub_g)
    # print(norm)
    sub_g.ndata['norm'] = torch.Tensor(norm).view(-1, 1)

    node_id = sub_g.ndata['id'].view(-1).tolist()
    sub_g.ids.update(zip(node_id, list(range(sub_g.number_of_nodes()))))
    return sub_g

def get_sorted_s_r_embed_rgcn(s_hist_data, s, t, graph_dict, seq_len, time_gap, time_start):
    s_hist = s_hist_data[0]
    s_hist_t = s_hist_data[1]
    # print(s_hist, s_hist_t)
    # print('---------------------------------')
    # print('s_hist', s_hist)
    # print('s_hist_t', s_hist_t)
    # print('---------------------------------')
    s_hist_len = torch.LongTensor(list(map(len, s_hist_t))).cuda()
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]
    s_hist_sorted = []
    s_hist_t_sorted = []
    s_time_iterval_sorted = []
    s_time_month = []
    s_time_day = []
    s_time_week = []
    if num_non_zero:
        for i, idx in enumerate(s_idx):
            if i == num_non_zero:
                break
            s_hist_sorted.append(s_hist[idx])
            s_hist_t_sorted.append(s_hist_t[idx])
            target_time = [t] *len(s_hist_t[idx])
            # target_time=torch.cat(target_time,dim=-1)
            target_time = torch.Tensor(target_time).cuda()

            hist_tensor=torch.Tensor(s_hist_t[idx]).cuda()
            # print(target_time - hist_tensor)
            s_time_iterval_sorted.append((target_time - hist_tensor)/time_gap)
            # print((target_time - hist_tensor)/time_gap)
            month_sorted, day_sorted, week_sorted = get_month_day_hour_min(s_hist_t[idx], time_gap, time_start)
            s_time_month.append(torch.Tensor(month_sorted)) 
            s_time_day.append(torch.Tensor(day_sorted))
            s_time_week.append(torch.Tensor(week_sorted))
        s_tem = s[s_idx]
        # r_tem = r[s_idx]
   
        neighs_t, len_s, node_list, max_len = get_neighs_by_t(s_hist_sorted, s_hist_t_sorted)
        
        # subgs, len_subgs, t_subgs = get_subg_triples(s_hist_sorted, s_hist_t_sorted)
        # g_list = get_g_list(subgs, t_subgs, graph_dict)
        g_list, g_id_dict, nodes_num = get_g_list_id(neighs_t, graph_dict)
        # node_ids_graph = get_node_ids_to_g_id(s_hist_t_sorted, g_list, g_id_dict, node_list,
        #                                       max_len, all_nodes_num=nodes_num)
        node_ids_graph = get_node_ids_to_g_id(s_hist_sorted, s_hist_t_sorted, s_tem, g_list, g_id_dict, seq_len, nodes_num)
        # g_list = [g.to(torch.device('cuda:'+str(torch.cuda.current_device()))) for g in g_list]  
        # batched_graph = dgl.batch(g_list)
        # batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id']].view(-1, ent_embeds.shape[1])

        # move_dgl_to_cuda(batched_graph)
    else:
        g_list = []
        node_ids_graph = []
        
    # padding_mask for tf
    padding_mask = torch.ones(num_non_zero, seq_len+1).cuda()
    for i in range(num_non_zero):
        padding_mask[i, :s_len[i]+1] = 0
    return g_list, node_ids_graph, s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, s_time_week, padding_mask, s_len, s_idx
    # return s_len_non_zero, s_tem, r_tem, batched_graph, node_ids_graph
# s_len_non_zero, s_time_iterval_sorted, s_time_month, s_time_day, neigh_num_of_s, flat_s, s_len_sorted, s_idx


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)  # indices : [B, number entity]
    indices = torch.nonzero(indices == target.view(-1, 1))  # indices : [B, 2] 第一列递增， 第二列表示对应的答案实体id在每一行的位置
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        # print(triple)
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans.append(h.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    # uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        # r_to_e[rel+num_rels].add(src)
        # r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    """
    :param node_id: node id in the large graph
    :param num_rels: number of relation
    :param src: relabeled src id
    :param rel: original rel id
    :param dst: relabeled dst id
    :param use_cuda:
    :return:
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    # uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    # g.uniq_r = uniq_r
    # g.r_to_e = r_to_e
    # g.r_len = r_len
    # if use_cuda:
    #     g.to(gpu)
    #     g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g

def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
            # print(filter_score_batch)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method):
    hits = [1, 3, 10, 20, 30, 40]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    if method.startswith("filter"):
        print("MRR ({}): {:.6f}".format(method, mrr.item()))
        for hit in hits:
            avg_count = torch.mean((total_rank <= hit).float())
            print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    """

    :param m:
    :param edges:
    :return: union number in a graph
    """
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def append_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    # output_label_list = []
    # for all_ans in all_ans_list:
    #     output = []
    #     ans = []
    #     for e1 in all_ans.keys():
    #         for r in all_ans[e1].keys():
    #             output.append([e1, r])
    #             ans.append(list(all_ans[e1][r]))
    #     output = torch.from_numpy(np.array(output))
    #     output_label_list.append((output, ans))
    # return output_label_list
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list


def slide_list(snapshots, k=1):
    """
    :param k: padding K history for sequence stat
    :param snapshots: all snapshot
    :return:
    """
    k = k  # k=1 需要取长度k的历史，在加1长度的label
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14s_inhis', 'ICEWS14', "ICEWS14l", "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI"]:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0]])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def generate_cand(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    cand_ans = []
    for _ in range(len(test_triples)):
        index_list = [index for index in top_indices[_]]
        cand_ans.append(index_list)
    # 转化为numpy array
    cand_ans = np.array(cand_ans, dtype=int)
    return cand_ans


def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    # for _ in range(len(test_triples)):
    #     h, r = test_triples[_][0], test_triples[_][1]
    #     if (sorted_score[_][0]-sorted_score[_][1])/sorted_score[_][0] > 0.3:
    #         if r < num_rels:
    #             predict_triples.append([h, r, indices[_][0]])

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a
