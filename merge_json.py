#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : merge_json.py
# @Author: 罗锦文
# @Date  : 2023/4/8
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json

def merge():
    squad1 = json.load(open(sys.argv[1], encoding="utf-8"))
    squad2 = json.load(open(sys.argv[2], encoding="utf-8"))
    squad3 = json.load(open(sys.argv[3], encoding="utf-8"))
    for art in squad1['data']:
        art['type'] = '_zh'
    for art in squad2['data']:
        art['type'] = '_v1.1'
    for art in squad3['data']:
        art['type'] = '_v2.0'
    merge = {}
    merge['data'] = squad1['data'] + squad2['data'] + squad3['data']
    json.dump(merge, sys.stdout)

def merge2():
    squad1 = json.load(open(sys.argv[1], encoding="utf-8"))
    squad2 = json.load(open(sys.argv[2], encoding="utf-8"))
    for art in squad1['data']:
        art['type'] = '_zh'
    for art in squad2['data']:
        art['type'] = '_en'
    merge = {}
    merge['data'] = squad1['data'] + squad2['data']
    json.dump(merge, sys.stdout)

def trans():
    squad1 = json.load(open(sys.argv[1], encoding="utf-8"))
    squad2 = json.load(open(sys.argv[2], encoding="utf-8"))
    id2zh = {}
    for article in squad1["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                id2zh[qa["id"]] = qa["question"]
    id2en = {}
    for article in squad2["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                id2en[qa["id"]] = qa["question"]
    en2zhdoc = []
    for article in squad1["data"]:
        article['type'] = '_en2zhdoc'
        status = True
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["id"] in id2en:
                    qa["question"] = id2en[qa["id"]]
                else:
                    status = False
        if status == True:
            en2zhdoc.append(article)
    zh2endoc = []
    # print(id2zh)
    # print(id2en)
    for article in squad2["data"]:
        article['type'] = '_zh2endoc'
        status = True
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["id"] in id2zh:
                    qa["question"] = id2zh[qa["id"]]
                else:
                    status = False
        # if status == True:
        zh2endoc.append(article)
    merge = {}
    squad1 = json.load(open(sys.argv[1], encoding="utf-8"))
    squad2 = json.load(open(sys.argv[2], encoding="utf-8"))
    for art in squad1['data']:
        art['type'] = '_zh'
    for art in squad2['data']:
        art['type'] = '_en'
    merge['data'] = squad1['data'] + squad2['data'] + en2zhdoc + zh2endoc
    # merge['data'] = en2zhdoc
    json.dump(merge, indent=4, ensure_ascii=False, fp=sys.stdout)

if __name__ == "__main__":
    trans()
