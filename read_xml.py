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
    from xml.dom.minidom import parse
    import xml.dom.minidom

    # 使用minidom解析器打开XML文档
    DOMTree = xml.dom.minidom.parse("./data/data_type.xml")
    Data = DOMTree.documentElement
    if Data.hasAttribute("name"):
        print("name element : %s" % Data.getAttribute("name"))

        # 在集合中获取所有国家
    Countrys = Data.getElementsByTagName("country")

