#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluation.py
# @Author: 罗锦文
# @Date  : 2023/3/14
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs

def pair_all():
    ranklist = []
    groupsize = 8
    for line in sys.stdin:
        items = line.strip().split("\t")
        ranklist.append([items[1], items[2]])
    pairs = len(ranklist) // groupsize
    good = 0
    bad = 0

    for i in range(pairs):
        datas = ranklist[i*groupsize:(i+1)*groupsize]
        # datas.sort(key=lambda x: x[1], reverse=True)
        for j in range(len(datas)):
            for k in range(j+1,len(datas)):
                if datas[j][1] == datas[k][1]:
                    continue
                if datas[j][0] >= datas[k][0]:
                    if datas[j][1] >= datas[k][1]:
                        good += 1
                    else:
                        bad += 1
                else:
                    if datas[j][1] < datas[k][1]:
                        good += 1
                    else:
                        bad += 1
    print(good, bad, good/(good+bad))

def pair_top():
    ranklist = []
    groupsize = 8
    for line in sys.stdin:
        items = line.strip().split("\t")
        ranklist.append([items[1], items[2]])
    pairs = len(ranklist) // groupsize
    good = 0
    bad = 0

    for i in range(pairs):
        datas = ranklist[i*groupsize:(i+1)*groupsize]
        datas.sort(key=lambda x: x[1], reverse=True)
        left, level = datas[0][0], datas[0][1]
        rights = datas[1:]
        for logit, lvl in rights:
            if lvl != level:
                if logit < left:
                    good += 1
                else:
                    bad += 1
    print(good, bad, good/(good+bad))

if __name__ == "__main__":
    # pair_all()
    pair_top()
