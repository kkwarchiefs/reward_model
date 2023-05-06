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
from doc_index import RerankContext

def test():
    ins = pd.read_csv(sys.argv[1], header=None)
    name2res = defaultdict(list)
    search = SearchContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    for v in ins.values:
        text = open(v[2]).read()#.replace('\n', ' ').replace('\t', ' ').strip()
        query = v[0]
        search.refresh_data(text)
        reslist = search.get_query_context(query, top_k=3)
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
    search = RerankContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    for line in sys.stdin:
        v = line.strip().split('\t')
        text = v[0]
        query = v[1]
        search.refresh_data(text)
        reslist = search.get_query_context_colbert(query, top_k=3)
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
        print(sea, file=fout)

def test4():
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
        print(sea, file=fout)

def test2():
    for line in sys.stdin:
        items = line.strip().split('\t')
        text = open(items[1]).read().strip().replace('\n', '').replace('\t', '')
        print(text, items[0], sep='\t')


def test_qa():
    ins = QAContext()
    while True:
        path = input("请输入文件：")
        ins.refresh_data(open(path).read())
        while True:
            query = input("\n请输入问题, 输入exit结束：")
            if query=="exit":
                break
            ins.get_query_context(query)

def test_rerank():
    ins = SearchContext()
    while True:
        path = input("请输入文件：")
        ins.refresh_data(open(path).read())
        while True:
            query = input("\n请输入问题, 输入exit结束：")
            if query=="exit":
                break
            ins.get_query_context_move(query)

def test_newrank():
    search = RerankContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    ins = pd.read_csv(sys.argv[1], header=None)
    fout = open('auto.test', 'w')
    for v in ins.values:
        text = open(v[2]).read()#.replace('\n', ' ').replace('\t', ' ').strip()
        query = v[0]
        search.refresh_data(text)
        reslist = search.get_query_context_colbert(query, top_k=3)
        searchres = '||'.join(reslist)
        # search_old.refresh_data(text)
        # qares = '||'.join(search_old.get_query_context(query))
        aws = v[3].split('|')
        rel, sea, qas = 0, 0, 0
        for aw in aws:
            # rel += v[1].count(aw)
            sea += searchres.count(aw)
            # qas += qares.count(aw)
        # print(query, [rel, sea, qas], sea > qas,  sep='\t', file=fout)
        print(sea, file=fout)

def input_newrank():
    ins = RerankContext()
    ins.piece_len = 330
    while True:
        path = input("请输入文件：")
        ins.refresh_data(open(path).read())
        while True:
            query = input("\n请输入问题, 输入exit结束：")
            if query=="exit":
                break
            ins.get_query_context_colbert(query)

if __name__ == "__main__":
    test3()
