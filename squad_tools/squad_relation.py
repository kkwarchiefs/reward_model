# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.
import pickle

import numpy as np
import math
import faiss
import time
import random
import os.path
import math
import sys
import pickle as pkl
import multiprocessing
from os.path import join, isfile
from os import listdir
from collections import defaultdict
random.seed(8888)
import json
import datetime
d = 768# dimension
b_fkey = 'msmarco_passage'
q_fkey = 'squad_query'

root_dir = '/search/ai/jamsluo/passage_rank/all_doc_tools'
# doclist = ['document_2021092812', 'document_2021093012']
query_file = 'query_ids.pkl'
doc_file = 'msmarco_passage.pkl'


class SimTextSearch:

    def __init__(self):
        pass


    def load_feats(self):
        stime = time.time()
        query_ins = pickle.load(open(query_file, 'rb'))
        self.q_feats = np.array(list(query_ins.values()))
        self.q_feats_key = list(query_ins.keys())
        doc_ins = pickle.load(open(doc_file, 'rb'))
        self.base_feats = np.array(list(doc_ins.values()))
        self.base_feats_key = list(doc_ins.keys())
        print('pkl loaded, ', time.time() - stime,)
        print(self.base_feats.size,)
        print(len(self.base_feats_key))
        print(self.q_feats.size,)
        print(len(self.q_feats_key))


    def index_init(self, load_cache=False):
        idx_fname = 'index/'+ b_fkey+'.faiss'
        stime = time.time()
        if os.path.isfile(idx_fname) and load_cache:
            self.index = faiss.read_index(idx_fname)
            print('index readed cache, ', time.time() - stime)
            return 
        self.index = faiss.IndexHNSWFlat(d, 32)
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 32
        # nlist = 500
        # quantizer = faiss.IndexFlatL2(d)
        # self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        # self.index.train(self.base_feats)
        self.index = faiss.IndexIDMap(self.index)
        # self.index = faiss.IndexFlatL2(d)

        self.index.add_with_ids(self.base_feats, np.array(range(len(self.base_feats_key)), dtype=np.int64)) # add may be a bit slower as well
        # self.index.add(self.base_feats)
        faiss.write_index(self.index, idx_fname)
        print('index added, ', time.time() - stime)


    def index_init_flat(self, load_cache=False):
        idx_fname = 'index/'+ b_fkey+'.flat.faiss'
        stime = time.time()
        if os.path.isfile(idx_fname) and load_cache:
            self.index = faiss.read_index(idx_fname)
            print('index readed cache, ', time.time() - stime)
            return 
        self.index = faiss.IndexFlat(d)
        self.index = faiss.IndexIDMap(self.index)
        
        self.index.add_with_ids(self.base_feats, np.array(range(len(self.base_feats_key)))) # add may be a bit slower as well
        faiss.write_index(self.index, idx_fname)
        print('index added, ', time.time() - stime)


    def search(self, xq, k, score=None):
        stime = time.time()
        print('index query start, ', time.time() - stime)
        # self.index.nprobe = 30
        # limits, self.D, self.I = self.index.range_search(xq, k)
        self.D, self.I = self.index.search(xq, k)
        print('index query finished, ', time.time() - stime)

        print('--')
        # print(limits)
        # print(len(limits))
        print(self.I)
        print(self.D)

        print(self.I.dtype)

        dist_file = './base_data/' + b_fkey + '_dist.np'
        sim_key_file = './base_data/' + b_fkey + '_sim_key.np'
        self.D.tofile(dist_file)
        self.I.tofile(sim_key_file)

        records = {}
        for i, row in enumerate(self.I):
            proc_part_records = []
            k_q = self.q_feats_key[i]

            for j, idx_y in enumerate(row):
                if i == idx_y:
                    continue
                k_sel = self.base_feats_key[idx_y]
                dist = self.D[i][j]
                dist = math.sqrt(dist)
                # if dist > score:
                #     break
                # if (j+1) > 1000:
                #     break
                proc_part_records.append((k_sel, str(dist)))

            # if (i+1)%10000==0:
            #     print(i+1)
            # break
            records[k_q] = proc_part_records

        pkl.dump(records, open('search_res/' + b_fkey + '_' + q_fkey + '_top.pkl', 'wb'))
        print('dumped ')

        print('all query title:', len(records))

        fw = open('search_res/' + b_fkey + '_' + str(score) + '_retrieval.txt', 'w')
        for k, v in records.items():
            res = {_v[0] for _v in v if _v[0] != k}
            res = list(res)
            for _res in res:
                print(k, _res, sep='\t', file=fw)
        fw.close()

if __name__=="__main__":
    sim_text_search = SimTextSearch() 
    sim_text_search.load_feats()
    sim_text_search.index_init()
    #sim_text_search.index_init_flat(load_cache=True)
    # qry = []
    # for idx, (xq, fea) in enumerate(zip(sim_text_search.q_feats, sim_text_search.q_feats_title)):
    #     _time = fea.split('\t')[1]
    #     if _time > '1634918400' and _time < '1635004800':
    #         qry.append((xq, idx))
    # query = random.sample(qry, 500)
    # xqs, keys = [], []
    # for a, b in query:
    #     xqs.append(a)
    #     keys.append(b)
    # xqs = np.array(xqs)
    xqs = sim_text_search.q_feats
    # topk = 30.*30.
    # sim_text_search.range_search(xq, topk)
    topk = 50
    sim_text_search.search(xqs, topk)
    # topk = 4000
    # score = 30
    # sim_text_search.search_random(xqs, keys, topk, score)
    # sim_text_search.sim_text_analysis()
