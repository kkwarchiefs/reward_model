#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_rm_data.py
# @Author: 罗锦文
# @Date  : 2023/3/8
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import codecs
import pandas as pd
from collections import defaultdict

def read_user():
    ins = pd.read_csv(sys.argv[1])
    name2res = defaultdict(list)
    for v in ins.values:
        name2res[v[2]].append([v[1], json.loads(v[3])])
    for k, v in name2res.items():
        for pair in v:
            res = []
            for p in pair[1]:
                res.append((str((p['model_id'], p['name'], p['level'])), p['level']))
            res.sort(key=lambda x: x[1], reverse=True)
            print(k, pair[0], '\t'.join([a[0]for a in res]), sep='\t')

def make_predict():
    for line in sys.stdin:
        prompt, resps = line.strip().split('\t')
        response_list = json.loads(resps)
        for candidate in response_list:
            print(prompt, (candidate['name'].replace('\t', ''), ), candidate['level'], sep='\t')

def read_dev(filename):
    promptset = set()
    for line in open(filename):
        items = line.strip().split('\t')
        promptset.add(items[0])
    return promptset

if __name__ == "__main__":
    make_predict()
    # promptset = read_dev(sys.argv[1])
    # promptset = set()
    # for line in sys.stdin:
    #     items = line.strip().split('\t')
    #     if items[0] not in promptset:
    #         print(items[0].replace('\n',''), items[2], sep='\t')

