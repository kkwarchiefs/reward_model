#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : get_translate.py
# @Author: 罗锦文
# @Date  : 2023/3/31
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json
import hashlib

def get_md5(sign):
    instance = hashlib.md5()
    instance.update(sign.encode("utf-8"))
    return instance.hexdigest()

def hamless_line():
    for idx, line in enumerate(sys.stdin):
        ins = json.loads(line)
        chosen = ins['chosen']
        rejected = ins['rejected']
        for id, prompt in enumerate([chosen, rejected]):
            prompt_new = '翻译一下内容：' + prompt
            obj = {
                "id": get_md5(prompt),
                "prompt": prompt_new,
                "label": str(idx) + '_' + str(id),
                "length": 4096,
                "response": ""
            }
            print(json.dumps(obj, ensure_ascii=False))

def query_translations():
    res = []
    for line in sys.stdin:
        items = line.strip().split('\t')
        res.append(items)
    step = len(res) // 20
    for i in range(step+1):
        part = res[i*20: (i+1)*20]
        queries = []
        for id, a in enumerate(part):
            queries.append(str(id+1) + '.'+a[1])
        querystr = '\n'.join(queries)
        idx = [a[0] for a in part]
        prompt_new = '翻译一下内容：\n' + querystr
        obj = {
            "id": get_md5(prompt_new),
            "prompt": prompt_new,
            "label": str(idx),
            "length": int(4096 - len(prompt_new)),
            "response": ""
        }
        print(json.dumps(obj, ensure_ascii=False))

def query_translations_en():
    res = []
    for line in sys.stdin:
        ins = json.loads(line)
        res.append([ins["question_id"], ins["question"]])
    step = len(res) // 10
    for i in range(step+1):
        part = res[i*10: (i+1)*10]
        queries = []
        for id, a in enumerate(part):
            queries.append(str(id+1) + '.'+a[1])
        querystr = '\n'.join(queries)
        idx = [a[0] for a in part]
        prompt_new = 'translation to english:\n' + querystr
        obj = {
            "id": get_md5(prompt_new),
            "prompt": prompt_new,
            "label": str(idx),
            "length": int(4096 - 3*len(prompt_new)),
            "response": ""
        }
        print(json.dumps(obj, ensure_ascii=False))

if __name__ == "__main__":
    query_translations_en()
    # for idx, line in enumerate(sys.stdin):
    #     ins = json.loads(line)
    #     chosen = ins['chosen']
    #     rejected = ins['rejected']
    #     print(chosen.replace('\n', " "))
    #     print(rejected.replace('\n', " "))
