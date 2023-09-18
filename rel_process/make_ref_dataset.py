#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : make_ref_dataset.py
# @Author: 罗锦文
# @Date  : 2023/9/7
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json
import re
from collections import defaultdict
def show():
    key = 0
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
                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    print(key, {
                        "title": title,
                        "context": context,
                        "question": qa["question"],
                        "id": qa["id"] + suffix,
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answers,
                        },
                    })
                    key += 1

import hashlib

def get_md5(sign):
    instance = hashlib.md5()
    instance.update(sign.encode("utf-8"))
    return instance.hexdigest()

def find_match(left, right):
    wordset = set(left)
    index = []
    for i, word in enumerate(right):
        if word in wordset:
            index.append(i)
    return right[index[0]: index[-1]+1]

def make_data():
    for idx, line in enumerate(sys.stdin):
        ins = json.loads(line)
        response = ins['response']
        try:
            resp = json.loads(response)
            convert_sentence = resp['新句子']
        except:
            continue
        if len(convert_sentence) > 200 or len(convert_sentence) < len(ins["sentence"]) / 3 or len(convert_sentence) > len(ins["sentence"]) * 2:
            print(convert_sentence, '#####', ins["sentence"], file=sys.stderr)
            continue
        # try:
        #     text = find_match(convert_sentence, ins["sentence"])
        # except:
        #     print(convert_sentence, ins["sentence"], file=sys.stderr)
        #     continue
        text = ins["sentence"].strip()
        new_index = ins['context'].find(text)
        if new_index == -1:
            continue
        new = {
            'title': ins['context'][:10],
            "context": ins['context'],
            "question": convert_sentence,
            "id": get_md5(ins['context'] + str(idx)),
            "answers": {
                "answer_start": [new_index],
                "text": [text],
            },
        }
        print(json.dumps(new, ensure_ascii=False))

def filter_text(text):
    # 这个正则表达式匹配任何非中文、非英文、非数字的字符
    pattern = re.compile('[^\u4e00-\u9fa5a-zA-Z\d]')
    # 使用空字符串替换这些匹配的字符
    filtered_text = re.sub(pattern, '', text)
    return filtered_text


def __split_sentences(text):
    parts = re.split('[。？！\n；]', text)
    return [x for x in parts if len(filter_text(x)) > 10]

def make_pred():
    for a, b in zip(open(sys.argv[1]), open(sys.argv[2])):
        content = a.strip()
        start = content.find('根据上面内容回答问题')
        content = content[:start].replace('其中第1部分文章内容是：', '').replace('其中第2部分文章内容是：', '').replace('其中第3部分文章内容是：', '')
        query = b.strip()
        sentence = __split_sentences(query)
        if len(sentence) == 0:
            continue
        for sent in sentence:
            new = {
                'title': content[:10],
                "context": content,
                "question": sent,
                "id": get_md5(content + sent),
                "answers": {
                    "answer_start": [-1],
                    "text": [],
                },
            }

            print(json.dumps(new, ensure_ascii=False))

def make_compare():
    idx2query = {}
    idx2content = {}
    for line in open(sys.argv[1]):
        ins = json.loads(line)
        idx2query[ins['id']] = ins['question']
        idx2content[ins['id']] = ins['context']
    squad = json.load(open(sys.argv[2]))
    key2detail = {}
    for key, res in squad.items():
        filter = [a for a in res if len(a['text']) > 3]
        # key2detail[key] = filter
        sent = idx2query[key]
        content = idx2content[key]
        if len(filter) == 0:
            filter = ['NULL']
        res = [content, sent, str(filter[0])]
        res = [a.replace('\t','').replace('\n','<n>') for a in res]
        print('\t'.join(res))


def make_trans_data():
    prompt = """%s\nTranslate the above text into English:"""
    content2detail = {}
    for idx, line in enumerate(sys.stdin):
        ins = json.loads(line)
        response = ins['response']
        try:
            resp = json.loads(response)
            convert_sentence = resp['新句子']
        except:
            continue
        if len(convert_sentence) > 200:
            continue
        if ins['context'] not in content2detail:
            content2detail[ins['context']] = [(ins["sentence"], convert_sentence, ins['index'])]
        else:
            content2detail[ins['context']].append((ins["sentence"], convert_sentence, ins['index']))
    # print(len(content2detail))
    for key, value in content2detail.items():
        value.sort(key=lambda x:x[2])
        context = key
        last = 0
        part = 0
        for idx, tup in enumerate(value):
            start = context.find(tup[0])
            prefix = context[last: start]
            ins = {
                'context': prefix,
                'index': get_md5(context) + '\x01' + str(part),
                'sentences': value,
                'prompt': prompt % prefix,
            }
            print(json.dumps(ins, ensure_ascii=False))
            part += 1
            ins = {
                'context': tup[0],
                'index': get_md5(context) + '\x01' + str(part),
                'sentences': value,
                'prompt': prompt % tup[0],
            }
            print(json.dumps(ins, ensure_ascii=False))
            part += 1
            last = start + len(tup[0])
        if last < len(context) - 20:
            ins = {
                'context': context[last:],
                'index': get_md5(context) + '\x01' + str(part),
                'sentences': value,
                'prompt': prompt % context[last:],
            }
            print(json.dumps(ins, ensure_ascii=False))
            part += 1



if __name__ == "__main__":
    make_compare()
