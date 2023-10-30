#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : shuffle.py
# @Author: 罗锦文
# @Date  : 2023/9/13
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import random

if __name__ == "__main__":
    all_json = []
    for line in sys.stdin:
        all_json.append(line.strip())
    random.shuffle(all_json)
    for line in all_json:
        print(line)
