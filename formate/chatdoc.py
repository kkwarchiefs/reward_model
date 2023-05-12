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
from online_incontent import ReaderContext

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
        reslist = search.get_query_context(query, top_k=1)
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

def test_reader():
    ins = pd.read_csv(sys.argv[1], header=None)
    name2res = defaultdict(list)
    search = ReaderContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    for v in ins.values:
        text = open(v[2]).read()#.replace('\n', ' ').replace('\t', ' ').strip()
        query = v[0]
        search.refresh_data(text)
        reslist = search.get_query_context(query, top_k=1)
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
        text = open(v[1].strip()).read().replace('\n', ' ')
        query = v[0]
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

def test_online():
    name2res = defaultdict(list)
    search = SearchContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    for line in sys.stdin:
        v = line.strip().split('\t')
        text = open(v[1].strip()).read().replace('\n', ' ')
        query = v[0]
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

def test_move():
    name2res = defaultdict(list)
    search = SearchContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    for line in sys.stdin:
        v = line.strip().split('\t')
        text = open(v[1].strip()).read().replace('\n', ' ')
        query = v[0]
        search.refresh_data(text)
        reslist = search.get_query_context_move(query, top_k=3)
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
    ins.piece_len = 334
    while True:
        path = input("请输入文件：")
        ins.refresh_data(open(path).read())
        while True:
            query = input("\n请输入问题, 输入exit结束：")
            if query=="exit":
                break
            ins.get_query_context_colbert(query)

# def input_newrank():
#     ins = ReaderContext()
#     ins.piece_len = 334
#     while True:
#         path = input("请输入文件：")
#         ins.refresh_data(open(path).read())
#         while True:
#             query = input("\n请输入问题, 输入exit结束：")
#             if query=="exit":
#                 break
#             ins.get_query_context(query)


def test_sentence():
    name2res = defaultdict(list)
    search = SearchContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    ins = QAContext()
    for line in sys.stdin:
        v = line.strip().split('\t')
        text = open(v[1].strip()).read().replace('\n', ' ')
        query = v[0]
        search.refresh_data(text)
        reslist = search.get_query_context(query, top_k=20)
        newpara = []
        otherpara = []
        for para in reslist:
            ins.refresh_data(para)
            if ins.do_mrc(query):
                newpara.append(para)
            else:
                otherpara.append(para)
            if len(newpara) == 3:
                break
        if len(newpara) < 3:
            newpara += otherpara[:3-len(newpara)]
        searchres = '||'.join(newpara)
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

def test_mrc():
    name2res = defaultdict(list)
    search = QAContext()
    search.piece_len = 330
    # search_old = SearchContext()
    # search_old.model_name = "embedding_mul_onnx"
    fout = open('auto.test', 'w')
    for line in sys.stdin:
        v = line.strip().split('\t')
        text = open(v[1].strip()).read().replace('\n', ' ')
        query = v[0]
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

if __name__ == "__main__":
    test_move()
