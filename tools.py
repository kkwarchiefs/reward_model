#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tools.py
# @Author: 罗锦文
# @Date  : 2023/3/25
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs

if __name__ == "__main__":
    for line in sys.stdin:
        items = line.strip().split("\t")
        print(line.strip(), len(items[1])//50, sep='\t')
