#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : add_prefix.py
# @Author: 罗锦文
# @Date  : 2023/9/13
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import random
import json
import langid
import jieba
prefix = """这篇文章讲述了
在这篇文章中
根据文章的介绍
根据文章内容
据文章的描述
根据文章介绍
对于这篇文章中提及
根据文章提供的信息显示
根据文章提供的数据显示
该篇文章总结了
文章并未给出明确原因
文章提到
据提供的文章内容
根据文章所述
这篇文章报道了
这篇文章讨论了
这篇文章主要讲述了
根据文章所提供的内容
从文章中可以了解到
根据本文内容提及
这篇论文提到了
这本书的"""

def addprefix():

    all_json = []
    prefix_split = prefix.split('\n')
    for idx, line in enumerate(sys.stdin):
        if idx % 10 == 0:
            ins = json.loads(line)
            question = ins['question']
            head = random.choice(prefix_split)
            question = head + question
            ins['question'] = question
            all_json.append(json.dumps(ins, ensure_ascii=False))
        else:
            all_json.append(line.strip())
    random.shuffle(all_json)
    for line in all_json:
        print(line)

def longest_common_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)

    # 创建一个二维数组来保存最长公共子序列的长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 动态规划计算最长公共子序列的长度
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 返回最长公共子序列的长度
    return dp[m][n]

def similarity(str1, str2):
    lcs_length = longest_common_subsequence(str1, str2)
    similarity = lcs_length / len(str1)
    return similarity


prefix2 = """是的，
不是的，
对的，
不，
否，
是，
作者认为，
因为
根据文中描述，
不对
表面上，
首先，
根据本文，
据报道，
不一定，
"""

def rm_tokens():
    for idx, line in enumerate(sys.stdin):
        ins = json.loads(line)

        question = ins['question']
        answers = ins['answers']['text'][0]
        lang_id = langid.classify(answers)
        if lang_id[0] != 'zh':
            print(line.strip())
            continue
        if answers:
            both = set(question) & set(answers)
            score = len(both) / max(len(question), len(answers))
            if score < 0.25:
                continue
            if question[0] == answers[0]:
                if random.randint(1, 3) == 1:
                    words = jieba.lcut(question)
                    start = random.randint(1, 3)
                    question_new = ''.join(words[start:])
                    if len(question_new) > 10:
                        ins['question'] = question_new
                        print(json.dumps(ins, ensure_ascii=False))
                        continue
            if question[-1] == answers[-1]:
                if random.randint(1, 3) == 1:
                    words = jieba.lcut(question)
                    start = random.randint(1, 3)
                    question_new = ''.join(words[:-start])
                    if len(question_new) > 10:
                        ins['question'] = question_new
                        print(json.dumps(ins, ensure_ascii=False))
                        continue
        print(line.strip())


def addprefix2():
    all_json = []
    prefix_split = prefix2.split('\n')
    for idx, line in enumerate(sys.stdin):
        if idx % 10 == 0:
            ins = json.loads(line)
            question = ins['question']
            if '文章' in question:
                continue
            head = random.choice(prefix_split)
            question = head + question
            ins['question'] = question
            all_json.append(json.dumps(ins, ensure_ascii=False))
        else:
            all_json.append(line.strip())
    random.shuffle(all_json)
    for line in all_json:
        print(line)

if __name__ == "__main__":
    addprefix2()
