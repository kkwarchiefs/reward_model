#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tools.py
# @Author: 罗锦文
# @Date  : 2023/5/9
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
import codecs
import os
import random

from online_incontent import ReaderContext
import random
import hashlib

from transformers import AutoTokenizer
model_path = "/search/ai/pvopliu/glm_10m/GLM/GLM/convert_scripts/glm_10b_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
def get_md5(sign):
    instance = hashlib.md5()
    instance.update(sign.encode("utf-8"))
    return instance.hexdigest()

allpath = set()


def walkFile(file):
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            if f.endswith('.txt'):
                path = os.path.join(root, f)
                if path in allpath:
                    continue
                allpath.add(path)
                text = open(path).read()
                print(path, len(text), sep='\t')

        # 遍历所有的文件夹
        for d in dirs:
            walkFile(os.path.join(root, d))

def checkdata():
    fout1 = open(sys.argv[2], 'w')
    fout2 = open(sys.argv[3], 'w')
    for line in open(sys.argv[1]):
        context = line.strip()
        parts = ReaderContext.cut_doc_plus(context, piece_len=330)
        main_id = get_md5(context)
        print(main_id, context, sep='\t', file=fout1)
        for part in parts:
            print(main_id, get_md5(part), part, sep='\t', file=fout2)

def checknot(rsp, last, vlist):
    punctuation = '。？！，；、：、‘’“”（）/【】——.?!;,:\'()[]{}-'
    for x in rsp:
        if x in punctuation:
            return False
    for idx, part_doc in enumerate(vlist):
        if idx != last:
            if rsp in part_doc:
                return False
    return True

def create_query():
    main2part = collections.defaultdict(list)
    for line in sys.stdin:
        items = line.strip().split('\t')
        main2part[items[0]].append(items[2])
    for _, vlist in main2part.items():
        if len(vlist) < 3:
            continue
        random.shuffle(vlist)
        for idx, part_doc in enumerate(vlist[:20]):
            if len(part_doc) < 30:
                continue
            input_ids = tokenizer(part_doc, max_length=512, truncation=True, return_tensors="pt", padding=True)["input_ids"][0]
            status = True
            index = len(input_ids) // 10
            rsp = None
            count = 0
            while status:
                rstart = random.randint(0, index-1)*10
                rend = random.randint(4, 8)
                rsp = tokenizer.decode(input_ids[rstart: rstart+rend])
                if checknot(rsp, idx, vlist):
                    status = False
                else:
                    count += 1
                if count == 7:
                    status = False
            if count == 7:
                continue
            # print(rsp, vlist[idx], 1, sep='\t')
            # for other in vlist[idx+1:]:
            #     print(rsp, other, 0, sep='\t')
            rand_idx = random.randint(0, len(vlist)-1)
            while rand_idx == idx:
                rand_idx = random.randint(0, len(vlist) - 1)
            print(rsp, vlist[idx], vlist[rand_idx], sep='\t')


def checkdata2():
    fout1 = open(sys.argv[2], 'w')
    fout2 = open(sys.argv[3], 'w')
    for line in open(sys.argv[1]):
        path = line.strip().split('\t')[0]
        context = open(path).read().replace('\n', ' ').replace('\t', ' ')
        parts = ReaderContext.cut_doc_plus(context, piece_len=330)
        main_id = get_md5(context)
        print(main_id, context, sep='\t', file=fout1)
        for part in parts:
            print(main_id, get_md5(part), part, sep='\t', file=fout2)

def tools():
    for line in sys.stdin:
        ins = json.loads(line)
        response = ins['response']
        try:
            response = json.loads(response)
            if type(response[0]) != str:
                response = [a['question'] for a in response]
            response = [a.replace('<question>', '').replace('</question>', '') for a in response]
            print(ins['label'], response, sep='\t')
        except:
            pass

if __name__ == "__main__":
    import json
    main2part = collections.defaultdict(list)
    parts2main = {}
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        main2part[items[0]].append(items[2])
        parts2main[items[2]] = items[0]
    query2pos = {}
    for line in open(sys.argv[2]):
        items = line.strip().split('\t')

        query2pos[items[0]] = items[1]
    for line in sys.stdin:
        items = line.strip().split('\t')
        pos = eval(items[1])
        qset = set(items[0])
        newlist = []
        for p in pos:
            pset = set(p)
            if len(qset&pset) > 3:
                newlist.append(p)
        pos = query2pos.get(items[0])
        if pos:
            main_id = parts2main[pos]
            all_doc = main2part[main_id]
            for change in newlist[:len(all_doc)]:
                neg = random.choice(all_doc)
                while neg == pos:
                    neg = random.choice(all_doc)
                print(change, pos, neg, sep='\t')
