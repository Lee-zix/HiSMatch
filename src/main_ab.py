# @Time    : 2019-08-10 11:20
# @Author  : Lee_zix
# @Email   : Lee_zix@163.com
# @File    : main.py
# @Software: PyCharm
"""
The entry of the KGEvolve
"""

import argparse
import itertools
import os
import sys
import time
import pickle
import copy

import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import pre_packed_query_input, pre_packed_node_input, build_sub_graph
from src.hyperparameter_range import hp_range
from collections import defaultdict
from src.multievolve import MultiEvolve


def test_ablation(dev_start, node_list, dataset, exp, id2entity, id2relation, s_hist, o_hist, cands_history, all_ans_list):
    num_rel = len(id2relation.keys())
    with open("../data/"+dataset+"/scores.pkl", "rb") as f:
        scores_list, triples_list = pickle.load(f)
    with open("../data/"+dataset+"/scores_"+exp+ ".pkl", "rb") as f:
        scores_list1, triples_list = pickle.load(f)
    fout = open("../data/"+dataset+"/case_candidate.txt", "w")
    sample_num = 0
    for time_idx, test_triples in enumerate(tqdm(triples_list)): 
        scores = scores_list[time_idx]
        scores1 = scores_list1[time_idx]
        
        _, _, _, rank_filter = utils.get_total_rank(test_triples, scores, all_ans_list[time_idx], eval_bz=1000, rel_predict=0) 
        _, _, _, rank_filter1 = utils.get_total_rank(test_triples, scores1, all_ans_list[time_idx], eval_bz=1000, rel_predict=0) 
        
        his_begin = sum([len(_) for _ in triples_list[0: time_idx]])
        his_end = his_begin + len(triples_list[time_idx])
        his_begin = int(his_begin/2)
        his_end = int(his_end/2)

        s_history = [s_hist[0][_] for _ in range(his_begin, his_end)]
        s_history_t = [s_hist[1][_] for _ in range(his_begin, his_end)]
        o_history = [o_hist[0][_] for _ in range(his_begin, his_end)]
        o_history_t = [o_hist[1][_] for _ in range(his_begin, his_end)]

        long_history = s_history + o_history
        long_history_t = s_history_t + o_history_t

        hist = [cands_history[0][time_idx][_] for _ in node_list]
        hist_t = [cands_history[1][time_idx][_] for _ in node_list]

        for i in range(len(test_triples)):
            if i >= int(len(test_triples)/2):
                sample_num += 1
                continue
            h, r, t = test_triples[i, :]
            h, r, t = h.item(), r.item(), t.item()
            rank = rank_filter[i]
            rank1 = rank_filter1[i]
            if rank == 1 and rank1 != 1:
                scores1_sample = scores1[i, :]
                # print(scores1_sample)
                _, indices = torch.sort(scores1_sample, descending=True)
                ans_1 = indices[0].item()

                fout.write("time:{} query:{}:{}-{}-{}, rank: {} rank1: {}".format(dev_start+time_idx, sample_num, id2entity[h], id2relation[r], id2entity[t], rank, rank1))
                fout.write("\n")
                fout.write("=============query history==================\n")
                for q_his, q_his_t in zip(long_history[i][-5:], long_history_t[i][-5:]):
                    for rq, tq in q_his :    
                        fout.write("{}\t{}\t{}\n".format(q_his_t, id2relation[rq], id2entity[tq]))
                fout.write("=============history for ans {}================\n".format(id2entity[t]))
                for e_his, e_his_t in zip(hist[t], hist_t[t]):
                    for _, re, te in e_his:
                        if re >= num_rel:
                            fout.write("{}\t{}\t{}\n".format(e_his_t, "inv_"+id2relation[re-num_rel], id2entity[te]))
                        else:
                            fout.write("{}\t{}\t{}\n".format(e_his_t, id2relation[re], id2entity[te]))   
                
                fout.write("=============history for wrong {}================\n".format(id2entity[ans_1]))
                for e_his, e_his_t in zip(hist[ans_1][-5:], hist_t[ans_1][-5:]):
                    for _, re, te in e_his:
                        if re >= num_rel:
                            fout.write("{}\t{}\t{}\n".format(e_his_t, "inv_"+id2relation[re-num_rel], id2entity[te]))
                        else:
                            fout.write("{}\t{}\t{}\n".format(e_his_t, id2relation[re], id2entity[te]))    
            sample_num+=1     


def run_experiment(args):
    # load graph data
    data = utils.load_data(args.dataset)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    # Loading history data for each sample.
    history_data_dir = "../data/" + args.dataset
    with open(os.path.join(history_data_dir, 'dev_history_sub.txt'), 'rb') as f:
        s_history_dev_data = pickle.load(f)
    with open(os.path.join(history_data_dir, 'dev_history_ob.txt'), 'rb') as f:
        o_history_dev_data = pickle.load(f)

    # Loading answers for time-filterings
    all_ans_list_train, all_ans_list_valid, _ = utils.load_ans_for_eval(data, num_rels, num_nodes, ans_type=False)
    

    with open(os.path.join(history_data_dir, '{}_dev_ent_history_t.pkl'.format(str(args.phase))), 'rb') as f:
        dev_cand_list, dev_history_list, dev_history_t_list = pickle.load(f)

    all_node_id = torch.LongTensor([_ for _ in range(num_nodes)])

    test_ablation(
        len(all_ans_list_train)+args.start_time,
        all_node_id,
        args.dataset,
        "candidate",
        data.entity_dict,
        data.relation_dict,
        s_history_dev_data, 
        o_history_dev_data, 
        (dev_history_list, dev_history_t_list),
        all_ans_list_valid) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=7000,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", type=int, default=0,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--t-hidden", type=int, default=32,
                        help="dimension of time embeddings")                     
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--history-len", type=int, default=10,
                        help="history length")
    # parser.add_argument("--method",  type=int, default=0,
    #                     help="perform grid search for best configuration")
    parser.add_argument("--phase",  type=int, default=0,
                        help="phase number")
    parser.add_argument("--start-time",  type=int, default=1,
                        help="phase number")
    parser.add_argument("--time-gap",  type=int, default=1,
                        help="phase number")
    parser.add_argument("--topk",  type=int, default=20,
                        help=" number of candidates")
    parser.add_argument("--seq-model", type=str, default="gru",
                        help="opn of compgcn")
    parser.add_argument("--background-len", type=int, default=3,
                        help="background length")
    parser.add_argument("--bg-encoder", type=str, default="uvrgcn-sub",
                        help="opn of compgcn")

    args = parser.parse_args()
    run_experiment(args)



