#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : chatdoc.py
# @Author: 罗锦文
# @Date  : 2023/4/20
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import pandas as pd
from collections import defaultdict
from create_countext import *

def test():
    ins = pd.read_csv(sys.argv[1], header=None)
    name2res = defaultdict(list)
    search = SearchContext()
    search.piece_len = 400
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    for v in ins.values:
        text = open(v[2]).read()#.replace('\n', ' ').replace('\t', ' ').strip()
        query = v[0]
        search.refresh_data(text)
        reslist = search.get_query_context_move(query, top_k=3)
        searchres = '||'.join(reslist)
        # search_old.refresh_data(text)
        # qares = '||'.join(search_old.get_query_context(query))
        aws = v[3].split('|')
        rel, sea, qas = 0, 0, 0
        for aw in aws:
            rel += v[1].count(aw)
            sea += searchres.count(aw)
            # qas += qares.count(aw)
        # print(query, [rel, sea, qas], sea > qas,  sep='\t', file=fout)
        print(sea, file=fout)

def test3():
    name2res = defaultdict(list)
    search = SearchContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    for line in sys.stdin:
        v = line.strip().split('\t')
        text = v[0]
        query = v[1]
        search.refresh_data(text)
        reslist = search.get_query_context(query, top_k=3)
        searchres = '||'.join(reslist)
        # search_old.refresh_data(text)
        # qares = '||'.join(search_old.get_query_context(query))
        aws = v[2].split('|')
        rel, sea, qas = 0, 0, 0
        for aw in aws:
            # rel += v[1].count(aw)
            sea += searchres.count(aw)
            # qas += qares.count(aw)
        # print(query, [rel, sea, qas], sea > qas,  sep='\t', file=fout)
        print([rel, sea, qas], file=fout)

def test2():
    for line in sys.stdin:
        items = line.strip().split('\t')
        text = open(items[1]).read().strip().replace('\n', '').replace('\t', '')
        print(text, items[0], sep='\t')

if __name__ == "__main__":
    test()
