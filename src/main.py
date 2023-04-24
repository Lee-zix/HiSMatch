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


# os.environ['KMP_DUPLICATE_LIB_OK']='True'
def dilate_input(input_list):
    dilate_temp = input_list[0]

    for i in range(len(input_list)):
        if i == 0:
            continue
        else:
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
    dilate_temp = np.unique(dilate_temp, axis=0)
    return dilate_temp

def test_1nd(args, model, head_list, history_list, test_list, packed_query_input_list, packed_node_input_list, num_nodes, num_rels, all_ans_list, model_name, mode, generate_cand=False):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    idx = 0
    start_time = args.start_time + len(history_list) * args.time_gap
    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    if generate_cand:
        cand_query_list = []
        all_score_list = []

    all_node_id = torch.LongTensor([_ for _ in range(num_nodes)])
    input_list = [snap for snap in history_list[-args.background_len:]]
    # all_score_list = []
    # all_tri_list = []
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        # dilate the latest a few background graphs into a big graph.
        if args.background_len:
            input_snap = dilate_input(input_list) 
            history_g = build_sub_graph(num_nodes, num_rels, input_snap, True, args.gpu)
        else:
            history_g = None
       
        head_id = head_list[time_idx]
        test_triples_input = torch.LongTensor(test_snap).cuda()
        test_triples, final_score = model.predict(start_time+time_idx*args.time_gap, 
                                                    history_g,
                                                    test_triples_input,
                                                    head_id,
                                                    all_node_id,
                                                    packed_query_input_list[time_idx],
                                                    packed_node_input_list[time_idx])

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
        # all_score_list.append(final_score)
        # all_tri_list.append(test_triples)
        if generate_cand:
            cand_query = utils.generate_cand(test_triples, num_nodes, num_rels, final_score, args.topk)
            cand_query_list.append(cand_query)
            all_score_list.append(final_score)
        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    
    # with open("../data/"+args.dataset+"/scores.pkl", "wb") as f:
    #     pickle.dump((all_score_list, all_tri_list), f)

    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    if generate_cand:
        return cand_query_list, all_score_list
    else:
        return mrr_raw, mrr_filter


def run_experiment(args):
    # load graph data
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    args.batch_size = num_nodes if args.batch_size > num_nodes else args.batch_size

    # Loading history data for each sample.
    history_data_dir = "../data/" + args.dataset

    # Loading answers for time-filterings
    _, all_ans_list_valid, all_ans_list_test = utils.load_ans_for_eval(data, num_rels, num_nodes, ans_type=False)
    
    model_name = "{}-{}-ly{}-his{}-bghis{}-encoder{}-bgencoder{}-gpu{}-time{}-gap{}"\
        .format(args.dataset, args.seq_model, args.n_layers, args.history_len, args.background_len, args.encoder, args.bg_encoder, args.gpu, args.t_hidden,args.time_gap)

    with open(os.path.join(history_data_dir, 'train_graphs.txt'), 'rb') as f:
        graph_dict = pickle.load(f)


    model_name = "1-phase-" + model_name
    model_state_file = '../models/' + model_name
    model = MultiEvolve(args.time_gap,
                        args.start_time,
                        args.seq_model,
                        args.encoder,
                        num_nodes,
                        num_rels,
                        args.n_hidden,
                        args.t_hidden,
                        sequence_len=args.history_len,
                        dropout=args.dropout,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        bg_encoder=args.bg_encoder,
                        gpu = args.gpu,
                        graph_dict = graph_dict
                        )

    
    if args.phase:
        torch.cuda.set_device(args.gpu)
        model.cuda()   
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test==1 and os.path.exists(model_state_file):
        s_history_train_data, o_history_train_data, s_history_dev_data, o_history_dev_data, s_history_test_data, o_history_test_data = utils.load_his_data(history_data_dir, test=True)

        with open(os.path.join(history_data_dir, 'ent_history_dev.pkl'), 'rb') as f:
            dev_cand_list, dev_history_list, dev_history_t_list = pickle.load(f)
        with open(os.path.join(history_data_dir, 'ent_history_test.pkl'), 'rb') as f:
            test_cand_list, test_history_list, test_history_t_list = pickle.load(f)
        
        test_snap_list = [torch.from_numpy(snap).long().cuda() for snap in test_list]
        valid_snap_list = [torch.from_numpy(snap).long().cuda() for snap in valid_list]

        all_node_id = torch.LongTensor([_ for _ in range(num_nodes)])

        start_time = args.start_time + len(train_list) * args.time_gap
        packed_query_input_dev_list = pre_packed_query_input(valid_snap_list, s_history_dev_data, o_history_dev_data, start_time, args.time_gap, num_rels, args.history_len)
        start_time = args.start_time + len(train_list+valid_list) * args.time_gap
        packed_query_input_test_list = pre_packed_query_input(test_snap_list, s_history_test_data, o_history_test_data, start_time, args.time_gap, num_rels, args.history_len)
        

        train_head_list = [np.concatenate((snap[:, 0], snap[:, 2]), axis=0) for snap in train_list]
        dev_head_list = [np.concatenate((snap[:, 0], snap[:, 2]), axis=0) for snap in valid_list]
        test_head_list = [np.concatenate((snap[:, 0], snap[:, 2]), axis=0) for snap in test_list]
        start_time = args.start_time + len(train_list) * args.time_gap 
        packed_node_input_dev_list = pre_packed_node_input(valid_snap_list, all_node_id, (dev_history_list, dev_history_t_list), graph_dict, start_time, args.time_gap, num_rels, args.history_len)
        start_time = args.start_time + len(train_list+valid_list) * args.time_gap 
        packed_node_input_test_list = pre_packed_node_input(test_snap_list, all_node_id, (test_history_list, test_history_t_list), graph_dict, start_time, args.time_gap, num_rels, args.history_len)
    
        test_1nd(args,
            model,
            dev_head_list,
            train_list, 
            valid_list, 
            packed_query_input_dev_list,
            packed_node_input_dev_list,
            num_nodes,
            num_rels, 
            all_ans_list_valid, 
            model_state_file, 
            "test") 

        test_1nd(args,
                model,
                test_head_list,
                train_list+valid_list, 
                test_list, 
                packed_query_input_test_list, 
                packed_node_input_test_list,
                num_nodes, 
                num_rels,
                all_ans_list_test,
                model_state_file,
                mode="test")   
    
    elif args.test==0:
        s_history_train_data, o_history_train_data, s_history_dev_data, o_history_dev_data, s_history_test_data, o_history_test_data = utils.load_his_data(history_data_dir, test=False)

        with open(os.path.join(history_data_dir, 'ent_history_train.pkl'), 'rb') as f:
            _, train_history_list, train_history_t_list = pickle.load(f)
        with open(os.path.join(history_data_dir, 'ent_history_dev.pkl'), 'rb') as f:
            _, dev_history_list, dev_history_t_list = pickle.load(f)

        dev_head_list = [np.concatenate((snap[:, 0], snap[:, 2]), axis=0) for snap in valid_list]
     
        best_mrr = 0
        snap_list = [torch.from_numpy(snap).long().cuda() for snap in train_list]
        valid_snap_list = [torch.from_numpy(snap).long().cuda() for snap in valid_list]
        all_node_id = torch.LongTensor([_ for _ in range(num_nodes)])

        if os.path.exists("run_tmp_file_query_{}.pkl".format(args.dataset)) and os.path.exists("run_tmp_file_node_{}.pkl".format(args.dataset)):
            print("--------------------------------Loading packed historical graphs--------------------------------")
            packed_query_input_list, packed_query_input_dev_list = pickle.load(open("run_tmp_file_query_{}.pkl".format(args.dataset), "rb"))
            packed_node_input_list, packed_node_input_dev_list = pickle.load(open("run_tmp_file_node_{}.pkl".format(args.dataset), "rb"))
        else:
            print("--------------------------------Packing historical graphs at the first time to run the codes--------------------------------")
            packed_query_input_list = pre_packed_query_input(snap_list, s_history_train_data, o_history_train_data, args.start_time, args.time_gap, num_rels, args.history_len)
            packed_node_input_list = pre_packed_node_input(snap_list, all_node_id, (train_history_list, train_history_t_list), graph_dict, args.start_time, args.time_gap, num_rels, args.history_len)
        
            start_time = args.start_time + len(snap_list) * args.time_gap
            packed_query_input_dev_list = pre_packed_query_input(valid_snap_list, s_history_dev_data, o_history_dev_data, start_time, args.time_gap, num_rels, args.history_len)
            packed_node_input_dev_list = pre_packed_node_input(valid_snap_list, all_node_id, (dev_history_list, dev_history_t_list), graph_dict, start_time, args.time_gap, num_rels, args.history_len)
            with open("run_tmp_file_query_{}.pkl".format(args.dataset), "wb") as f:
                pickle.dump((packed_query_input_list, packed_query_input_dev_list), f)
            with open("run_tmp_file_node_{}.pkl".format(args.dataset), "wb") as f:
                pickle.dump((packed_node_input_list, packed_node_input_dev_list), f)
    
        print("------------------------------start training-----------------------------------------------")
        global_history_len = args.background_len
        for epoch in range(args.n_epochs):
            model.train()
            losses = [] 
            idx = [_ for _ in range(len(snap_list))]
            # random.shuffle(idx)
            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                if train_sample_num - global_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - global_history_len:
                                        train_sample_num]
                if global_history_len:
                    input_snap = dilate_input(input_list) 
                    history_g = build_sub_graph(num_nodes, num_rels, input_snap, True, args.gpu)
                else:
                    history_g = None

                ab_time = args.start_time+train_sample_num*args.time_gap
                output = snap_list[train_sample_num]

                # inverse_triples = output[:, [2, 1, 0]]
                # inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                # all_triples = torch.cat([output, inverse_triples])
                # all_triples = all_triples.to(args.gpu)
                
                packed_query_input = packed_query_input_list[train_sample_num]
                packed_node_input = packed_node_input_list[train_sample_num]
                assert(args.batch_size >= num_nodes)
               
                node_id = all_node_id
                head_list = torch.cat([output[:, 0], output[:, 2]])
                label_list = torch.cat([output[:, 2], output[:, 0]])

                loss = model.get_loss(ab_time,
                                    history_g,
                                    output, 
                                    head_list,
                                    node_id,
                                    packed_query_input, 
                                    packed_node_input,
                                    label_list.cuda())
                
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f}  Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test_1nd(args,
                                                model,
                                                dev_head_list,
                                                train_list, 
                                                valid_list,
                                                packed_query_input_dev_list,
                                                packed_node_input_dev_list,
                                                num_nodes, 
                                                num_rels,
                                                all_ans_list_valid,
                                                model_state_file,
                                                mode="train")
                if mrr_filter < best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = mrr_filter
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=80000,
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
    parser.add_argument("--phase",  type=int, default=1,
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



