#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : shuffle.py.py
# @Author: 罗锦文
# @Date  : 2023/2/22
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import random

if __name__ == "__main__":
    res = []
    for line in sys.stdin:
        res.append(line.strip())
    random.shuffle(res)
    for a in res:
        print(a)
