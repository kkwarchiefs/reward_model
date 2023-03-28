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
import random
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

def read_file(file_name):
    prompt2gpt = {}
    for line in open(file_name):
        ins = json.loads(line)
        prompt2gpt[ins["prompt"]] = ins["response"]
    return prompt2gpt

def make_new_sort():
    chat = read_file('./reward_data/chat_0317_all.json')
    res = []
    for line in sys.stdin:
        try:
            prompt, resps = line.strip().split('\t')
        except:
            print(line, file=sys.stderr)
            continue
        response_list = json.loads(resps)
        good = chat.get(prompt)
        if good:
            response_list.append({
                'name': good,
                'level': 9
            })
            res.append((prompt, json.dumps(response_list, ensure_ascii=False)))
        else:
            res.append((prompt, resps))
    random.shuffle(res)
    for k, v in res:
        print(k, v, sep='\t')

def read_rm_datas():
    for line in sys.stdin:
        ins = json.loads(line)
        status = False
        for tk in ['建议选择B',  'B更好', "\n\nB好"]:
            if tk in ins['response']:
                print( (ins['response'], ), 1, sep='\t')
                status = True
                break
        if status:
            continue
        status = False
        for tk in ['建议选择A', 'A更好', "\n\nA好"]:
            if tk in ins['response']:
                print((ins['response'],), 1, sep='\t')
                status = True
                continue
        if status:
            continue
        print((ins['response'],), file=sys.stderr)

def check_gpt():
    for line in sys.stdin:
        ins = json.loads(line)
        print(ins['prompt'], '"' + ins['best'].replace('"', "'") + '"', '"' + ins['bad'].replace('<n>', "\n").replace('"', "'") + '"', sep='\t')

def check_rsp():
    for prompt, rsp in zip(open('./reward_data/top_5000_25000.csv'), open('./reward_data/toolgpt_0.6_5000_25000.txt')):
        prompt = prompt.strip()
        rsp = rsp.strip()
        if '<n>' in prompt:
            prompt = '"' + prompt.replace('<n>', "\n").replace('"', "'") + '"'
        # if rsp.count('<n>') > 5:
        #     continue
        if '<n>' in rsp:
            rsp = '"' + rsp.replace('<n>', "\n").replace('"', "'") + '"'
        print(prompt, rsp.replace('<|startofpiece|>', ''), sep='\t')


def check_rsp2():
    for line in sys.stdin:
        items = line.strip().split('\t')
        rsp = '"' + items[0].replace('<n>', "\n").replace('[UNUSED1]', "\n").replace('"', "'") + '"'
        print(rsp, '\t'.join(items[1:]), sep='\t')

def read_set():
    prompt2res = {}
    for prompt, rsp in zip(open('./top_5000_25000.csv'), open('./toolgpt_0.6_5000_25000.txt')):
        prompt = prompt.strip()
        rsp = rsp.strip()
        if '<n>' in prompt:
            prompt = '"' + prompt.replace('<n>', "\n").replace('"', "'") + '"'
        if rsp.count('<n>') < 5:
            continue
        if '<n>' in rsp:
            rsp = '"' + rsp.replace('<n>', "\n").replace('"', "'") + '"'
        # prompt2res[prompt] = rsp.replace('<|startofpiece|>', '')
        print(prompt, rsp.replace('<|startofpiece|>', ''), sep='\t')
    # for line in sys.stdin:
    #     items = line.strip().split('\t')
    #     rsp = items[1]
    #     if items[0] in prompt2res:
    #         if '<n>' in rsp:
    #             rsp = '"' + rsp.replace('<n>', "\n").replace('"', "'") + '"'
    #         print(items[0], prompt2res[items[0]], rsp, sep='\t')

def prepare_train():
    import os
    for root, dirs, files in os.walk('./third_model/'):
        for file in files:
            if file.endswith('csv'):
                ins = pd.read_csv(root + '/' + file)
                for idx, v in enumerate(ins.values):
                    # if v[2] == -1:
                    #     continue
                    # if v[2] != 0 and v[2] != 1:
                    #     print(idx, v[2], file=sys.stderr)
                    #     continue
                    outs = v[0] + '[UNUSED1]' + v[1].replace('\n', '<n>')
                    print(outs.replace('\t', ''), 0, sep='\t')

def prepare_train2():
    ins = pd.read_csv(sys.argv[1])
    for idx, v in enumerate(ins.values):
        if v[2] == -1:
            continue
        if v[2] != 0 and v[2] != 1:
            print(idx, v[2], file=sys.stderr)
            continue
        rsp = v[1].strip().strip('"').replace("<|endoftext|>", "")
        outs = v[0] + '[UNUSED1]' + rsp.replace('\n', '<n>')

        print(outs.replace('\t', ''), int(v[2]), sep='\t')

def prepare_train3():
    ins = pd.read_csv(sys.argv[1])
    for v in ins.values:
        prompt = v[0].replace('[CLS]', '').replace('<|startofpiece|>', '').replace('[gMASK]', '').replace('[回答]', '').replace(' ', '')
        out_res = []
        for rsp in v[1:]:
            rsp = rsp.strip().strip('"').replace("<|endoftext|>", "").replace("<|endofpiece|>", '')
            outs = prompt + '[UNUSED1]' + rsp.replace('\n', '<n>')
            out_res.append(outs.replace('\t', ' '))
        print('\t'.join(out_res))

def chat_format():
    for line in sys.stdin:
        ins = json.loads(line)
        rsp = ins["response"]
        if '\n' in ins["response"]:
            rsp = '"' + rsp.replace('"', "'") + '"'
        print(ins["prompt"], rsp, sep='\t')

def read_user_sec():
    # ins = pd.read_csv(sys.argv[1])
    name2res = defaultdict(list)
    for line in open(sys.argv[1]):
        v = line.strip().split('\t')
        name2res[v[0]].append([v[1], json.loads(v[2])])
    for k, v in name2res.items():
        for pair in v:
            res = []
            for p in pair[1]:
                res.append((p['name'], p['level']))
            res.sort(key=lambda x: x[1], reverse=True)
            rsp = res[-1][0]
            if '\n' in rsp:
                rsp = '"' + rsp.replace('"', "'") + '"'
            print(k, rsp, sep='\t')

def chat_format2():
    for line in sys.stdin:
        line = line.strip().replace('<n>', '\n')
        rsp = line
        if '\n' in line:
            rsp = '"' + rsp.replace('"', "'") + '"'
        print(rsp, sep='\t')

def chat_format3():
    for line in sys.stdin:
        line = line.strip().replace('<n>', '\n')
        items = line.split('\t')
        newitem = []
        for item in items:
            if '\n' in item:
                newitem.append('"' + item.replace('"', "'") + '"')
            else:
                newitem.append(item)
        print('\t'.join(newitem), sep='\t')

def chat_rm_score():
    for line in sys.stdin:
        ins = json.loads(line)
        rsp = ins["response"]
        print(ins["prompt"].replace('\n', '<n>'), rsp.replace('\n', '<n>'), sep='\t')

if __name__ == "__main__":
    chat_format3()

