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
if __name__ == "__main__":
    ins = pd.read_csv(sys.argv[1])
    name2res = defaultdict(list)
    for v in ins.values:
        name2res[v[2]].append([v[1], json.loads(v[3])])
    model_ratio = defaultdict(dict)
    model_score = defaultdict(int)
    count = 0
    for k, v in name2res.items():
        for pair in v:
            count += 1
            res = []
            for p in pair[1]:
                res.append((str((p['model_id'], p['name'], p['level'])), p['level']))
                ins = model_ratio[p['model_id']]
                if p['level'] not in ins:
                    ins[p['level']] = 1
                else:
                    ins[p['level']] += 1
                model_score[p['model_id']] += p['level']
    for k,v in model_score.items():
        print(k, v/count)
    for k, v in model_ratio.items():
        print(k, {a: b/count for a,b in v.items()})
