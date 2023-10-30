#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : merge_trans.py
# @Author: 罗锦文
# @Date  : 2023/9/8
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json
from collections import defaultdict
# import nltk
import hashlib

def get_md5(sign):
    instance = hashlib.md5()
    instance.update(sign.encode("utf-8"))
    return instance.hexdigest()

# 下载punkt分词模型
# nltk.download('punkt')



if __name__ == "__main__":
    # 注意需要判断> 1/3 小于2倍
    queryall =set()
    idx2detail = defaultdict(list)
    for line in sys.stdin:
        ins = json.loads(line)
        tup = ins['index'].split('\x01')
        idx = tup[0]
        xx = int(tup[1])
        recent = (ins['context'], ins['response'], ins['sentences'], xx)
        idx2detail[idx].append(recent)
    for key, detail in idx2detail.items():
        detail.sort(key=lambda x:x[-1])
        last = 0
        origin = ''
        trans = ''
        res = []
        sent2query = {a[0]:a[1] for a in detail[0][2]}
        for part in detail:
            sent = part[0]
            if sent in sent2query:
                query = sent2query[sent]
                trans_sent = part[1]
                tokens = trans_sent.split(' ')
                if len(tokens) < len(sent)/4 or len(tokens) > len(sent)*2:
                    print((query, sent, part[1]), file=sys.stderr)
                    continue
                res.append((query, sent, part[1], len(origin), len(trans), len(trans)+len(part[1])))
            origin += part[0]
            trans += part[1]
        # print(json.dumps({
        #     'origin': origin,
        #     'trans': trans,
        #     'res': res
        # }, ensure_ascii=False))
        for tups in res:
            if tups[0] not in queryall:
                new = {
                    'title': trans[:30],
                    "context": trans,
                    "question": tups[0],
                    "id": get_md5(trans + tups[0]),
                    "answers": {
                        "answer_start": [tups[4]],
                        "text": [tups[2]],
                    },
                }
                print(json.dumps(new, ensure_ascii=False))
            queryall.add(tups[0])

