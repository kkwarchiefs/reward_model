#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_html.py
# @Author: 罗锦文
# @Date  : 2023/4/1
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
from bs4 import BeautifulSoup

if __name__ == "__main__":
    # w = open('./data/doc_type.html').readlines()
    bs = BeautifulSoup(open('./data/doc_type.html'), 'html.parser')
