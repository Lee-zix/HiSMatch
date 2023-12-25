# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 9:45 上午
# @Author  : Lee_zix
# @Email   : Lee_zix@163.com
# @File    : get_history.py
# @Software: PyCharm

from pandas.io import api
import numpy as np
import os
from collections import defaultdict
import pickle
import argparse
import dgl
import torch
from utils import get_total_number, load_quadruples


parser = argparse.ArgumentParser("preprocess")
parser.add_argument('-d', '--dataset', default="GDELT")
parser.add_argument('-his', '--history_len', type=int, default=5)
parser.add_argument('-dir', '--data_dir', type=str, default="../data/")

args = parser.parse_args()
data_path = os.path.join(args.data_dir, args.dataset)

train_data, train_times = load_quadruples('', os.path.join(data_path, "train.txt"))
test_data, test_times = load_quadruples('', os.path.join(data_path, "test.txt"))
dev_data, dev_times = load_quadruples('', os.path.join(data_path, "valid.txt"))
num_e, num_r = get_total_number(data_path, 'stat.txt')

total_data = np.concatenate([train_data, dev_data, test_data])
total_times = np.concatenate([train_times, dev_times, test_times])

# graph_dict_train = {}
# for tim in total_times:
#     print(str(tim)+'\t'+str(max(total_times)))
#     data = get_data_with_t(total_data, tim)
#     graph_dict_train[tim] = get_big_graph(data, num_r)

# with open(os.path.join(data_path, "train_graphs.txt"), 'wb') as fp:
    # pickle.dump(graph_dict_train, fp)


history_len = args.history_len

s_his = [[[] for _ in range(num_e)] for _ in range(num_r)]
o_his = [[[] for _ in range(num_e)] for _ in range(num_r)]
s_his_t = [[[] for _ in range(num_e)] for _ in range(num_r)]
o_his_t = [[[] for _ in range(num_e)] for _ in range(num_r)]

s_history_data = [[] for _ in range(len(train_data))]
o_history_data = [[] for _ in range(len(train_data))]
s_history_data_t = [[] for _ in range(len(train_data))]
o_history_data_t = [[] for _ in range(len(train_data))]

e = []
r = []
latest_t = 0

s_his_cache = [[[] for _ in range(num_e)] for _ in range(num_r)]
o_his_cache = [[[] for _ in range(num_e)] for _ in range(num_r)]
s_his_cache_t = [[[] for _ in range(num_e)] for _ in range(num_r)]
o_his_cache_t = [[[] for _ in range(num_e)] for _ in range(num_r)]

for i, train in enumerate(train_data):
    if i % 10000==0:
        print("train",i, len(train_data))
    t = train[3]
    if latest_t != t:
        for rr in range(num_r):
            for ee in range(num_e):
                if len(s_his_cache[rr][ee]) != 0:
                    if len(s_his[rr][ee]) >= history_len:
                        s_his[rr][ee].pop(0)
                        s_his_t[rr][ee].pop(0)
                    s_his[rr][ee].append(s_his_cache[rr][ee].copy())
                    s_his_t[rr][ee].append(s_his_cache_t[rr][ee].copy())
                    s_his_cache[rr][ee]= []
                    s_his_cache_t[rr][ee] = []
                if len(o_his_cache[rr][ee]) != 0:
                    if len(o_his[rr][ee]) >=history_len:
                        o_his[rr][ee].pop(0)
                        o_his_t[rr][ee].pop(0)
                    o_his[rr][ee].append(o_his_cache[rr][ee].copy())
                    o_his_t[rr][ee].append(o_his_cache_t[rr][ee].copy())
                    o_his_cache[rr][ee]=[]
                    o_his_cache_t[rr][ee]=[]
        latest_t = t
    s = train[0]
    r = train[1]
    o = train[2]
    # print(s_his[r][s])
    s_history_data[i] = s_his[r][s].copy()
    o_history_data[i] = o_his[r][o].copy()
    s_history_data_t[i] = s_his_t[r][s].copy()
    o_history_data_t[i] = o_his_t[r][o].copy()
    # s_his_cache[r][s].append([r, o])
    # o_his_cache[r][o].append([r, o])
    if len(s_his_cache[r][s]) == 0:
        s_his_cache[r][s] = np.array([[r, o]])
    else:
        s_his_cache[r][s] = np.concatenate((s_his_cache[r][s], [[r, o]]), axis=0)
    s_his_cache_t[r][s] = t
    if len(o_his_cache[r][o]) == 0:
        o_his_cache[r][o] = np.array([[r, s]])
    else:
        o_his_cache[r][o] = np.concatenate((o_his_cache[r][o], [[r, s]]), axis=0)
    o_his_cache_t[r][o] = t
    # print(s_history_data[i])
    # print(i)
    # print("hist",s_history_data[i])
    # print(s_his_cache[r][s])
    # print(s_history_data[i])

    # print(s_his_cache[r][s])
# with open('train_history_sub.txt', 'wb') as fp:
#     pickle.dump(s_history_data, fp)
# with open('train_history_ob.txt', 'wb') as fp:
#     pickle.dump(o_history_data, fp)
with open(os.path.join(data_path, "train_history_sub.txt"), 'wb') as fp:
    pickle.dump((s_history_data, s_history_data_t), fp)
with open(os.path.join(data_path, "train_history_ob.txt"), 'wb') as fp:
    pickle.dump((o_history_data, o_history_data_t), fp)

# print(s_history_data[0])
s_history_data_dev = [[] for _ in range(len(dev_data))]
o_history_data_dev = [[] for _ in range(len(dev_data))]
s_history_data_t_dev = [[] for _ in range(len(dev_data))]
o_history_data_t_dev = [[] for _ in range(len(dev_data))]


for i, dev in enumerate(dev_data):
    if i % 10000 ==0:
        print("valid",i, len(dev_data))
    t = dev[3]
    if latest_t != t:
        for rr in range(num_r):
            for ee in range(num_e):
                if len(s_his_cache[rr][ee]) != 0:
                    if len(s_his[rr][ee]) >= history_len:
                        s_his[rr][ee].pop(0)
                        s_his_t[rr][ee].pop(0)
                    s_his[rr][ee].append(s_his_cache[rr][ee].copy())
                    s_his_t[rr][ee].append(s_his_cache_t[rr][ee].copy())
                    s_his_cache[rr][ee] = []
                    s_his_cache_t[rr][ee] = []
                if len(o_his_cache[rr][ee]) != 0:
                    if len(o_his[rr][ee]) >=history_len:
                        o_his[rr][ee].pop(0)
                        o_his_t[rr][ee].pop(0)
                    o_his[rr][ee].append(o_his_cache[rr][ee].copy())
                    o_his_t[rr][ee].append(o_his_cache_t[rr][ee].copy())
                    o_his_cache[rr][ee] = []
                    o_his_cache_t[rr][ee] = []
        latest_t = t
    s = dev[0]
    r = dev[1]
    o = dev[2]
    s_history_data_dev[i] = s_his[r][s].copy()
    o_history_data_dev[i] = o_his[r][o].copy()
    s_history_data_t_dev[i] = s_his_t[r][s].copy()
    o_history_data_t_dev[i] = o_his_t[r][o].copy()
    # s_his_cache[r][s].append([r, o])
    # o_his_cache[r][o].append([r, s])
    if len(s_his_cache[r][s]) == 0:
        s_his_cache[r][s] = np.array([[r, o]])
    else:
        s_his_cache[r][s] = np.concatenate((s_his_cache[r][s], [[r, o]]), axis=0)
    s_his_cache_t[r][s] = t
    if len(o_his_cache[r][o]) == 0:
        o_his_cache[r][o] = np.array([[r, s]])
    else:
        o_his_cache[r][o] = np.concatenate((o_his_cache[r][o], [[r, s]]), axis=0)
    o_his_cache_t[r][o] = t


with open(os.path.join(data_path, "dev_history_sub.txt"), 'wb') as fp:
    pickle.dump((s_history_data_dev, s_history_data_t_dev), fp)
with open(os.path.join(data_path, "dev_history_ob.txt"), 'wb') as fp:
    pickle.dump((o_history_data_dev, o_history_data_t_dev), fp)

s_history_data_test = [[] for _ in range(len(test_data))]
o_history_data_test = [[] for _ in range(len(test_data))]
s_history_data_t_test = [[] for _ in range(len(test_data))]
o_history_data_t_test = [[] for _ in range(len(test_data))]

for i, test in enumerate(test_data):
    if i % 10000 ==0:
        print("test",i, len(test_data))
    t = test[3]
    if latest_t != t:
        for rr in range(num_r):
            for ee in range(num_e):
                if len(s_his_cache[rr][ee]) != 0:
                    if len(s_his[rr][ee]) >= history_len:
                        s_his[rr][ee].pop(0)
                        s_his_t[rr][ee].pop(0)
                    s_his[rr][ee].append(s_his_cache[rr][ee].copy())
                    s_his_t[rr][ee].append(s_his_cache_t[rr][ee].copy())
                    s_his_cache[rr][ee]= []
                    s_his_cache_t[rr][ee] = []
                if len(o_his_cache[rr][ee]) != 0:
                    if len(o_his[rr][ee]) >=history_len:
                        o_his[rr][ee].pop(0)
                        o_his_t[rr][ee].pop(0)
                    o_his[rr][ee].append(o_his_cache[rr][ee].copy())
                    o_his_t[rr][ee].append(o_his_cache[rr][ee].copy())
                    o_his_cache[rr][ee]=[]
                    o_his_cache_t[rr][ee]= []
        latest_t = t
    s = test[0]
    r = test[1]
    o = test[2]
    s_history_data_test[i] = s_his[r][s].copy()
    o_history_data_test[i] = o_his[r][o].copy()
    s_history_data_t_test[i] = s_his_t[r][s].copy()
    o_history_data_t_test[i] = s_his_t[r][o].copy()
    if len(s_his_cache[r][s]) == 0:
        s_his_cache[r][s] = np.array([[r, o]])
    else:
        s_his_cache[r][s] = np.concatenate((s_his_cache[r][s], [[r, o]]), axis=0)
    s_his_cache_t[r][s] = t
    if len(o_his_cache[r][o]) == 0:
        o_his_cache[r][o] = np.array([[r, s]])
    else:
        o_his_cache[r][o] = np.concatenate((o_his_cache[r][o], [[r, s]]), axis=0)
    o_his_cache_t[r][o] = t

with open(os.path.join(data_path, "test_history_sub.txt"), 'wb') as fp:
    pickle.dump((s_history_data_test, s_history_data_t_test), fp)
with open(os.path.join(data_path, "test_history_ob.txt"), 'wb') as fp:
    pickle.dump((o_history_data_test, o_history_data_t_test), fp)


