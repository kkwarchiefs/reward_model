#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : merge_json.py
# @Author: 罗锦文
# @Date  : 2023/4/8
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import codecs
import json
import uuid

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

def convert_cmrc():
    cmrc = json.load(open(sys.argv[1], encoding="utf-8"))
    cmrc_dev = json.load(open(sys.argv[2], encoding="utf-8"))
    # print(len(cmrc["data"]))
    key2qry = []
    for article in cmrc["data"] + cmrc_dev["data"][:500]:
        assert len(article["paragraphs"]) == 1
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                # if len(qa["answers"]) == 0:
                #     print(paragraph)
                key2qry.append((qa['id'], qa["question"], qa["trans_question"]))
    cmrc_multi = []
    for article in cmrc["data"] + cmrc_dev["data"][:500]:
        atc = {}
        atc['type'] = "_cmrc"
        atc['paragraphs'] = []
        atc['title'] = article['title']
        atc['id'] = article['id']
        for paragraph in article["paragraphs"]:
            zh = {
                'qas': [],
                'id': paragraph['id'],
                'context': paragraph['context'],

            }
            en = {
                'qas': [],
                'id': paragraph['id'],
                'context': paragraph['trans_context'],

            }
            zh2en = {
                'qas': [],
                'id': paragraph['id'],
                'context': paragraph['trans_context'],

            }
            en2zh = {
                'qas': [],
                'id': paragraph['id'],
                'context': paragraph['context'],

            }
            for qa in paragraph["qas"]:
                aws_zh = []
                aws_en = []
                for aws in qa["answers"]:
                    aws_zh.append({
                        "text": aws["text"],
                        "answer_start": aws["answer_start"]
                    })
                    aws_en.append({
                        "text": aws["trans_aligned_text"],
                        "answer_start": aws["trans_aligned_start"]
                    })
                qzh = {
                    "question": qa["question"],
                    "id": qa["id"] + "_zh",
                    "answers": aws_zh,
                    "is_impossible": False,
                }
                zh['qas'].append(qzh)
                qen = {
                    "question": qa["trans_question"],
                    "id": qa["id"] + "_en",
                    "answers": aws_en,
                    "is_impossible": False,
                }
                en['qas'].append(qen)
                qzh2en = {
                    "question": qa["question"],
                    "id": qa["id"] + "_zh2en",
                    "answers": aws_en,
                    "is_impossible": False,
                }
                zh2en['qas'].append(qzh2en)
                qen2zh = {
                    "question": qa["trans_question"],
                    "id": qa["id"] + "_en2zh",
                    "answers": aws_zh,
                    "is_impossible": False,
                }
                en2zh['qas'].append(qen2zh)
            no_anwers = random.choice(key2qry)
            if article['id'] not in no_anwers[0]:
                qzh = {
                    "question": no_anwers[1],
                    "id": no_anwers[0] + "_zh_" + uuid.uuid4().__str__(),
                    "answers": [],
                    "is_impossible": True,
                }
                zh['qas'].append(qzh)
            no_anwers = random.choice(key2qry)
            if article['id'] not in no_anwers[0]:
                qen = {
                    "question": no_anwers[2],
                    "id": no_anwers[0] + "_en_" + uuid.uuid4().__str__(),
                    "answers": [],
                    "is_impossible": True,
                }
                en['qas'].append(qen)
            no_anwers = random.choice(key2qry)
            if article['id'] not in no_anwers[0]:
                qzh2en = {
                    "question": no_anwers[1],
                    "id": no_anwers[0] + "_zh2en_" + uuid.uuid4().__str__(),
                    "answers": [],
                    "is_impossible": True,
                }
                zh2en['qas'].append(qzh2en)
            no_anwers = random.choice(key2qry)
            if article['id'] not in no_anwers[0]:
                qen2zh = {
                    "question": qa["trans_question"],
                    "id": no_anwers[0] + "_en2zh_" + uuid.uuid4().__str__(),
                    "answers": [],
                    "is_impossible": True,
                }
                en2zh['qas'].append(qen2zh)
            atc['paragraphs'].extend([zh, en, en2zh, zh2en])
        cmrc_multi.append(atc)
    multi = {
        'version': "cmrc_multi",
        'data': cmrc_multi,
    }
    json.dump(multi, indent=4, ensure_ascii=False, fp=sys.stdout)


def merge3():
    squad1 = json.load(open(sys.argv[1], encoding="utf-8"))
    squad2 = json.load(open(sys.argv[2], encoding="utf-8"))
    # paras = [len(a) for a in squad1['data']]
    # print(sum(paras), len(squad2['data'][0]['paragraphs']))
    merge = {}
    merge['data'] = squad1['data'] + squad2['data']
    json.dump(merge, sys.stdout)

def merge4():
    squad1 = json.load(open(sys.argv[1], encoding="utf-8"))
    squad2 = json.load(open(sys.argv[2], encoding="utf-8"))
    squad3 = json.load(open(sys.argv[2], encoding="utf-8"))
    # paras = [len(a) for a in squad1['data']]
    # print(sum(paras), len(squad2['data'][0]['paragraphs']))
    merge = {}
    merge['data'] = squad1['data'] + squad2['data'] + squad3['data']
    json.dump(merge, sys.stdout)


if __name__ == "__main__":
    merge4()
