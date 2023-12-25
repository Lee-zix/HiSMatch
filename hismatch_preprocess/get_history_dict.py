from tqdm import tqdm
import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
import torch
import argparse
import sys
sys.path.append("../")
from utils import get_total_number, get_data_with_t

def load_quadruples(num_rels, inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            quadrupleList.append([tail, rel+num_rels, head, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                quadrupleList.append([tail, rel+num_rels, head, time]) 
                times.add(time)
    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)

def add_history(timestamps, triplet_list, init_start_time, start_time, time_gap, num_rels, num_nodes, cand_ans, history_dict, history_t_dict):
    candidate_list = []
    history_list = []
    history_t_list = []

    # print("len(triplets_list) is {}".format(len(triplets_list)))
    for retime, timestamp_idx in enumerate(tqdm(timestamps)):
        triplets = get_data_with_t(triplet_list, timestamp_idx)
        triplets = np.array(triplets)
        # print(triplets)
        # print("retime is {}".format(retime))
        time = time_gap*retime + start_time
        # print("time is {}".format(time))
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



def get_big_graph(data, num_rels):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    # src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    # rel_o = np.concatenate((rel + num_rels, rel))
    # rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    # g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type'] = torch.LongTensor(rel)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g

parser = argparse.ArgumentParser("preprocess")
parser.add_argument('-d', '--dataset', default="GDELT")
parser.add_argument('-his', '--history-len', type=int, default=5)
parser.add_argument('-start-time', '--start-time', type=int, default=1)
parser.add_argument('-gap', '--gap', type=int, default=1)
parser.add_argument('-dir', '--data_dir', type=str, default="../data/")

args = parser.parse_args()
data_path = os.path.join(args.data_dir, args.dataset)
history_len = args.history_len
# data = load_data(args.data_dir, args.dataset)
# train_list = split_by_time(data.train)
# valid_list = split_by_time(data.valid)
# test_list = split_by_time(data.test)
# all_list = train_list + valid_list + test_list
num_e, num_r = get_total_number(data_path, 'stat.txt')

train_data, train_times = load_quadruples(num_r, '', os.path.join(data_path, "train.txt"))
test_data, test_times = load_quadruples(num_r, '', os.path.join(data_path, "test.txt"))
dev_data, dev_times = load_quadruples(num_r, '', os.path.join(data_path, "valid.txt"))

all_history_dict = {}
all_history_t_dict = {}

s_his = [[] for _ in range(num_e)]
o_his = [[] for _ in range(num_e)]
s_his_t = [[] for _ in range(num_e)]
o_his_t = [[] for _ in range(num_e)]
e = []
r = []
latest_t = 0
s_his_cache = [[] for _ in range(num_e)]
o_his_cache = [[] for _ in range(num_e)]
s_his_cache_t = [None for _ in range(num_e)]
o_his_cache_t = [None for _ in range(num_e)]


def search_history(search_data, latest_t, mode="train"):
    print("{} samples in {}".format(len(search_data), mode))
    s_history_data = [[] for _ in range(len(search_data))]
    o_history_data = [[] for _ in range(len(search_data))]
    s_history_data_t = [[] for _ in range(len(search_data))]
    o_history_data_t = [[] for _ in range(len(search_data))]
    
    for i, train in enumerate(tqdm(search_data)):
        # if i % 10000 == 0:
        #     print(mode, i, len(search_data))
        t = train[3]
        if latest_t != t:
            for ee in range(num_e):
                if len(s_his_cache[ee]) != 0:
                    if len(s_his[ee]) >= history_len:
                        s_his[ee].pop(0)
                        s_his_t[ee].pop(0)

                    s_his[ee].append(s_his_cache[ee].copy())
                    s_his_t[ee].append(s_his_cache_t[ee])
                    s_his_cache[ee] = []
                    s_his_cache_t[ee] = None
                if len(o_his_cache[ee]) != 0:
                    if len(o_his[ee]) >= history_len:
                        o_his[ee].pop(0)
                        o_his_t[ee].pop(0)

                    o_his[ee].append(o_his_cache[ee].copy())
                    o_his_t[ee].append(o_his_cache_t[ee])
                    o_his_cache[ee] = []
                    o_his_cache_t[ee] = None
            latest_t = t
        s = train[0]
        r = train[1]
        o = train[2]

        s_history_data[i] = s_his[s].copy()
        o_history_data[i] = o_his[o].copy()
        s_history_data_t[i] = s_his_t[s].copy()
        o_history_data_t[i] = o_his_t[o].copy()
        # print(o_history_data_g[i])
        all_history_dict[(s, t)] = s_his[s].copy()
        all_history_dict[(o, t)] = o_his[o].copy()
        all_history_t_dict[(s, t)] = s_his_t[s].copy()
        all_history_t_dict[(o, t)] = o_his_t[o].copy()

        if len(s_his_cache[s]) == 0:
            s_his_cache[s] = np.array([[s, r, o]])
        else:
            s_his_cache[s] = np.concatenate((s_his_cache[s], [[s, r, o]]), axis=0)
        s_his_cache_t[s] = t

        if len(o_his_cache[o]) == 0:
            o_his_cache[o] = np.array([[o, r, s]])
        else:
            o_his_cache[o] = np.concatenate((o_his_cache[o], [[o, r, s]]), axis=0)
        o_his_cache_t[o] = t

    return s_history_data, o_history_data, s_history_data_t, o_history_data_t, latest_t


s_history_data, o_history_data, s_history_data_t, o_history_data_t, latest_t = search_history(train_data, latest_t, "train")
s_history_data_dev, o_history_data_dev, s_history_data_dev_t, o_history_data_dev_t, latest_t = search_history(dev_data, latest_t, "dev")
s_history_data_test, o_history_data_test, s_history_data_test_t, o_history_data_test_t, _ = search_history(test_data, latest_t, "test")


with open(os.path.join(data_path, 'all_history_dict.pkl'), 'wb') as fp:
    pickle.dump(all_history_dict, fp)
with open(os.path.join(data_path, 'all_history_t_dict.pkl'), 'wb') as fp:
    pickle.dump(all_history_t_dict, fp)
with open(os.path.join(data_path, 'all_history_dict.pkl'), 'rb') as fp:
    all_history_dict = pickle.load(fp)
with open(os.path.join(data_path, 'all_history_t_dict.pkl'), 'rb') as fp:
    all_history_t_dict = pickle.load(fp)


start_time = args.start_time
train_cand_list, train_history_list, train_history_t_list = add_history(train_times, train_data, args.start_time ,start_time, args.gap, num_r, num_e, None, all_history_dict, all_history_t_dict)

start_time = len(train_times) * args.gap + args.start_time
dev_cand_list, dev_history_list, dev_history_t_list = add_history(dev_times, dev_data, args.start_time, start_time, args.gap, num_r, num_e, None, all_history_dict, all_history_t_dict)    

start_time = (len(train_times) + len(dev_times)) * args.gap + args.start_time
test_cand_list, test_history_list, test_history_t_list = add_history(test_times, test_data, args.start_time, start_time, args.gap, num_r, num_e, None, all_history_dict, all_history_t_dict)  

with open(os.path.join(data_path, 'ent_history_train.pkl'), 'wb') as f:
    pickle.dump((train_cand_list, train_history_list, train_history_t_list), f)
with open(os.path.join(data_path, 'ent_history_dev.pkl'), 'wb') as f:
    pickle.dump((dev_cand_list, dev_history_list, dev_history_t_list), f)
with open(os.path.join(data_path, 'ent_history_test.pkl'), 'wb') as f:
    pickle.dump((test_cand_list, test_history_list, test_history_t_list), f)


