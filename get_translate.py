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

if __name__ == "__main__":
    for idx, line in enumerate(sys.stdin):
        ins = json.loads(line)
        chosen = ins['chosen']
        rejected = ins['rejected']
        print(chosen.replace('\n', " "))
        print(rejected.replace('\n', " "))
