#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_json.py
# @Author: 罗锦文
# @Date  : 2023/4/9
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import codecs
import json
from collections import defaultdict

def old():
    ins = json.load(open(sys.argv[1]))
    pred = ins[0]
    refer = ins[1]
    id2detail = defaultdict(dict)
    for a in pred:
        id2detail[a["id"]]["pred"] = a["prediction_text"]
    for a in refer:
        id2detail[a["id"]]["refer"] = a["answers"]
    for k, v in id2detail.items():
        print(k, v["pred"], v["refer"], sep='\t')

def train():
    count = 0
    all = 0
    with open(sys.argv[1], encoding="utf-8") as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "")
            para_list = []
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].replace("\n", " ")  # do not strip leading blank spaces GH-2585
                para_list.append(context)
                qa_pos = []
                for qa in paragraph["qas"]:
                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    if len(answer_starts) > 0:
                        answers = [answer["text"] for answer in qa["answers"]]
                        question = qa["question"].replace("\n", " ")
                        qa_pos.append([question, context, answer_starts[0], answers[0]])
            topk = 20 if len(para_list) > 20 else len(para_list)
            for pair in qa_pos:
                negs = random.sample(para_list, k=topk)
                for txt in negs:
                    if txt != pair[1]:
                        print(pair[0], pair[1], txt, sep='\t')
            count += 1
            all += len(para_list)
    print(all/ count, file=sys.stderr)

def dev():
    with open(sys.argv[1], encoding="utf-8") as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "")
            para_list = []
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].replace("\n", " ")  # do not strip leading blank spaces GH-2585
                para_list.append(context)
                qa_pos = []
                for qa in paragraph["qas"]:
                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    if len(answer_starts) > 0:
                        answers = [answer["text"] for answer in qa["answers"]]
                        question = qa["question"].replace("\n", " ")
                        qa_pos.append([question, context, answer_starts[0], answers[0]])
            topk = 9 if len(para_list) > 9 else len(para_list)
            for pair in qa_pos:
                negs = random.sample(para_list, k=topk)
                print(pair[0], pair[1], 1, sep='\t')
                for txt in negs:
                    if txt != pair[1]:
                        print(pair[0], txt, 0, sep='\t')

if __name__ == "__main__":
    train()
    # dev()
    # for line in sys.stdin:
    #     items = line.strip().split("\t")
    #     if len(items) != 3:
    #         print(line.strip())

