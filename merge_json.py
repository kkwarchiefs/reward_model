#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : merge_json.py
# @Author: 罗锦文
# @Date  : 2023/4/8
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json

if __name__ == "__main__":
    squad1 = json.load(open(sys.argv[1], encoding="utf-8"))
    squad2 = json.load(open(sys.argv[2], encoding="utf-8"))
    merge = {}
    merge['data'] = squad1['data'] + squad2['data']
    json.dump(merge, sys.stdout)
