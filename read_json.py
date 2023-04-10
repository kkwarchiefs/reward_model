#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_json.py
# @Author: 罗锦文
# @Date  : 2023/4/9
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

if __name__ == "__main__":
    predins = json.load(open(sys.argv[2], encoding="utf-8"))
    with open(sys.argv[1], encoding="utf-8") as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "")
            suffix = ''
            if 'type' in article:
                suffix = article['type']
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                for qa in paragraph["qas"]:
                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]
                    question =  qa["question"]
                    id =  qa["id"] + suffix
                    answers = {
                        "answer_start": answer_starts,
                        "text": answers,
                    }
                    pred = predins[id]
                    print(id, question, pred, answers,sep='\t')
