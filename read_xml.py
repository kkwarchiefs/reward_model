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

import numpy as np
from bs4 import BeautifulSoup
import re
import xml.etree.cElementTree as ET
from collections import defaultdict

import torch
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer
import langid
from typing import Optional, Tuple
import collections
import json
from bm25_search import *
from create_countext import QAContext
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
model_path = "/search/ai/pretrain_models/chatglm-6b/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

class PaperContext():
    def __init__(self):
        self.model_name = "embedding_pooling_onnx"  # 模型目录名/venus注册模型名称
        address = "10.212.207.33:8000"  # 机器地址
        self.triton_client = httpclient.InferenceServerClient(url=address)
        rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
        self.tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
        self.max_token_length = 1000

    def refresh_data(self, filename):
        self.paper = Paper()
        self.paper.read_xml(filename)
        self.paper.split_document()
        part_docs = [a.__repr__() for a in self.paper.subsections]
        self.subsections_embedding = self.get_embedding(part_docs)
        para = defaultdict(list)
        for i in range(len(self.subsections_embedding)):
            j = self.paper.subindex[i]
            para[j].append(self.subsections_embedding[i])
            # assert self.paper.subsections[i] in self.paper.section_lines[j].__repr__()
            # logging.debug('#####'.join([self.paper.subsections[i], self.paper.section_lines[j].__repr__()]))
        temp_embedding = []
        sort_para = sorted(para.items(), key=lambda a:a[0])
        for _, v in sort_para:
            temp_embedding.append(v)
        assert len(temp_embedding) == len(self.paper.section_lines)
        top_dict = defaultdict(list)
        self.para_embedding = []
        for sec, emb in zip(self.paper.section_lines, temp_embedding):
            top_dict[sec.section_one].extend(emb)
            mean_embedding = np.mean(emb, axis=0)
            self.para_embedding.append(mean_embedding)
            sec.embedding = mean_embedding
        self.para_embedding = np.array(self.para_embedding)
        self.top_embedding = []
        assert len(top_dict) == len(self.paper.section_top)
        for k in self.paper.section_top:
            emb_list = top_dict[k]
            self.top_embedding.append(np.mean(emb_list, axis=0))
        self.top_embedding = np.array(self.top_embedding)
        # print(self.subsections_embedding.shape)
        # print(self.para_embedding.shape)
        # print(self.top_embedding.shape)
        # print(np.array(self.para_embedding).size, len(self.paper.section_lines))
        # print(np.array(self.top_embedding).size)

    def get_embedding(self, doc):
        RM_input = self.tokenizer(doc, max_length=512, truncation=True, return_tensors="pt", padding=True)
        # print(RM_input)
        RM_batch = [torch.tensor(RM_input["input_ids"]).numpy(), torch.tensor(RM_input["attention_mask"]).numpy()]

        inputs = []
        inputs.append(httpclient.InferInput('input_ids', list(RM_batch[0].shape), 'INT64'))
        inputs.append(httpclient.InferInput('attention_mask', list(RM_batch[1].shape), 'INT64'))
        inputs[0].set_data_from_numpy(RM_batch[0])
        inputs[1].set_data_from_numpy(RM_batch[1])
        output = httpclient.InferRequestedOutput('output')
        # try:
        results = self.triton_client.infer(
            self.model_name,
            inputs,
            model_version='1',
            outputs=[output],
            request_id='1'
        )
        results = results.as_numpy('output')
        return results

    @staticmethod
    def cut_doc_mean(data, piece_len=400):
        tokens = tokenizer(data)
        tokens_ids = tokens['input_ids'][1:-1]
        steps = len(tokens_ids) // piece_len + 1
        mean_len = len(tokens_ids) // steps + 1
        index = 0
        piece_data = []
        last_index = 0
        while index < len(tokens_ids):
            index += mean_len
            if index < len(tokens_ids):
                temp_data = tokenizer.decode(tokens_ids[last_index: index])
            else:
                temp_data = tokenizer.decode(tokens_ids[last_index:])
            temp_len = min(index, len(tokens_ids)) - last_index
            piece_data.append((temp_data, temp_len))
            last_index = index
        return piece_data

    def get_query_context(self, query, top_k=3):
        mrc = QAContext()
        query_emb = self.get_embedding(query)
        scores = np.matmul(
            query_emb,
            self.subsections_embedding.transpose(1, 0))[0]
        index_list = np.argsort(scores)[::-1]
        max_len = self.max_token_length
        context = []
        print("="*20)
        print(query)
        parent_set = set()
        for index in index_list:
            sub_para = self.paper.subsections[index]
            # mrc.refresh_data(self.paper.subsections[index])
            #
            # reslist = mrc.get_query_context(query)
            # print(reslist)
            # parent = self.paper.subindex[index]
            # neighbours = self.paper.subindex_reverse[parent]
            # print([scores[a] for a in neighbours])
            if max_len > 0:
                # mrc.refresh_data(self.paper.subsections[index])
                # reslist = mrc.do_mrc(query)
                # if len(reslist) == 0:
                #     print("=====empty=====", self.paper.subsections[index], scores[index], self.paper.sub_length[index], sep='###')
                #     continue
                print(sub_para.context, scores[index], sub_para.para_length, sep='###')
                parent = self.paper.subindex[index]
                if parent in parent_set:
                    continue
                neighbours = self.paper.subindex_reverse[parent]
                print([scores[a] for a in neighbours])
                if len(neighbours) > 1:
                    mrc.refresh_data(self.paper.section_lines[parent].__repr__())
                    reslist = mrc.do_mrc(query)
                    status = False
                    for a in reslist:
                        if a[0] not in list(range(sub_para.start_position[0], sub_para.start_position[1])):
                            status = True
                    print(reslist)
                    if status:
                        context.append(self.paper.section_lines[parent].__repr__())
                        parent_set.add(parent)
                        max_len -= sum([self.paper.subsections[a].para_length for a in neighbours])
                        continue
                context.append(sub_para.__repr__())
                max_len -= sub_para.para_length
        return context

    def get_query_context_debug(self, query, top_k=3):
        query_emb = self.get_embedding(query)
        scores = np.matmul(
            query_emb,
            self.subsections_embedding.transpose(1, 0))[0]
        context = []
        index_list = np.argsort(scores)[::-1]
        for index in index_list[:top_k]:
            print(self.paper.subsections[index], scores[index], self.paper.sub_length[index], sep='###')
            parent = self.paper.subindex[index]
            neighbours = self.paper.subindex_reverse[parent]
            print([scores[a] for a in neighbours])
        print('#'*30)
        scores = np.matmul(
            query_emb,
            self.para_embedding.transpose(1, 0))[0]
        context = []
        index_list = np.argsort(scores)[::-1]
        for index in index_list[:top_k]:
            print(self.paper.section_lines[index], scores[index])
        print('#' * 30)
        scores = np.matmul(
            query_emb,
            self.top_embedding.transpose(1, 0))[0]
        context = []
        index_list = np.argsort(scores)[::-1]
        for index in index_list:
            print(self.paper.section_top[index], scores[index])
        return context


class Paragraph:
    def __init__(self, section_one, section_two, position, context, start_position=None, para_length=0):
        self.section_one = section_one
        self.section_two = section_two
        self.context = context
        self.position = position
        self.start_position = start_position
        self.para_length = para_length
        self.embedding = None

    def __repr__(self):
        return ' '.join([self.section_one, self.section_two, self.context])


class Paper:
    def __init__(self):
        # 初始化函数，根据pdf路径初始化Paper对象
        self.section_names = []   # 段落标题
        self.section_top = []
        self.section_texts = {}   # 段落内容
        self.section_lines = []
        self.abstract = ''
        self.title = ''
        self.section_tree = {}
        self.subindex = {}
        self.subindex_reverse = {}
        self.subsections = []
        # self.sub_length = []
        self.part_length = 500

    def split_document(self):
        for idx, para in enumerate(self.section_lines):
            parts = PaperContext.cut_doc_mean(para.context, self.part_length)
            begin = len(self.subsections)
            start_at = 0
            for part in parts:
                self.subindex[len(self.subsections)] = idx
                # self.subsections.append(part)
                sub_para = Paragraph(para.section_one, para.section_two, para.position, part[0], (start_at, len(part)), part[1])
                start_at += len(part)
                self.subsections.append(sub_para)
                # self.sub_length.append(part[1])
            self.subindex_reverse[idx] = list(range(begin, len(self.subsections)))

    @staticmethod
    def replace_string(xmlstring):
        xmlstring = re.sub(' xmlns="[^"]+"', '', xmlstring)
        xmlstring = re.sub('<ref[^>]+>[^>]+</ref>', '', xmlstring)
        # xmlstring = re.sub('<ref [^>]+>', '', xmlstring)
        # xmlstring = re.sub('</ref>', '', xmlstring)
        xmlstring = re.sub('<formula [^>]+>', '', xmlstring)
        xmlstring = re.sub('</formula>', '', xmlstring)
        xmlstring = re.sub('<label [^>]+>', '', xmlstring)
        xmlstring = re.sub('<label>', '', xmlstring)
        xmlstring = re.sub('</label>', '', xmlstring)
        xmlstring = re.sub('<cell>', '', xmlstring)
        xmlstring = re.sub('<cell [^>]+>', '', xmlstring)
        xmlstring = re.sub('</cell>', ' ', xmlstring)
        xmlstring = re.sub('<row>', '', xmlstring)
        xmlstring = re.sub('</row>', ' ', xmlstring)
        return xmlstring

    def read_xml(self, path):
        xmlstring = open(path).read()
        xmlstring = Paper.replace_string(xmlstring)
        # print(xmlstring)
        # 使用minidom解析器打开XML文档
        DOMTree = ET.ElementTree(ET.fromstring(xmlstring))
        self.title = DOMTree.find('teiHeader/fileDesc/titleStmt/title').text
        self.abstract = DOMTree.find('teiHeader/profileDesc/abstract/div/p').text
        self.section_names.append('Abstract')
        # para = Paragraph('Abstract', '', '1', self.abstract)
        # self.section_lines.append(para)
        # self.section_top.append('Abstract')
        # self.section_tree['Abstract'] = []
        # self.section_texts[self.abstract] = len(self.section_names)
        body = DOMTree.find('text/body')

        last_content = ''
        last_section = ''
        position = ''
        for part in body.iter('div'):
            subline = list(part)
            position = subline[0].attrib.get('coords', 'NULL')
            if 'n' in subline[0].attrib:
                if last_content:
                    self.section_texts[last_content] = len(self.section_names)
                    if last_section ==  self.section_names[-1]:
                        para = Paragraph(last_section, '', position, last_content)
                    else:
                        para = Paragraph(last_section, self.section_names[-1], position, last_content)
                    self.section_lines.append(para)
                    last_content = ''
                section = subline[0].attrib['n']
                if section.count('.') == 0:
                    self.section_top.append(subline[0].text)
                    self.section_tree[subline[0].text] = []
                    last_section = subline[0].text
                else:
                    self.section_tree[last_section].append(subline[0].text)
                self.section_names.append(subline[0].text)
                last_content += " ".join([a.text for a in subline[1:]])
            else:
                last_content += " ".join([a.text for a in subline])
        if last_content:
            self.section_texts[last_content] = len(self.section_names)
            if last_section == self.section_names[-1]:
                para = Paragraph(last_section, '', position, last_content)
            else:
                para = Paragraph(last_section, self.section_names[-1], position, last_content)
            self.section_lines.append(para)
        for part in body.iter('figure'):
            position = part.attrib.get('coords', 'NULL')
            section = part.attrib['{http://www.w3.org/XML/1998/namespace}id']
            if "fig" in section:
                continue
            # print(list(part))
            content = ' '.join([a.text for a in list(part) if a.text])
            para = Paragraph(section, '', position, content)
            self.section_lines.append(para)
            self.section_top.append(section)
            self.section_tree[section] = []
            self.section_texts[content] = len(self.section_names)
        # print(len(self.section_top))
        # print(self.section_top)
        # print(len(self.section_names))
        # print(self.section_names)
        # print(self.section_texts.keys())
        # # print(self.section_tree)
        # print(len(self.section_texts))
        # # for k, v in self.section_texts.items():
        # #     print(k)
        # #     print(v)
        # for k in self.section_lines:
        #     print(k)

def get_lines():
    fout = open('test.txt', 'w')
    for line in sys.stdin:
        ins = Paper()
        items = line.strip().split('\t')
        ins.read_xml(items[1])
        text = ''
        for a in ins.section_lines:
            text += a.__repr__()
        print(text, items[0], items[2], sep='\t', file=fout)

def test_score():
    search = PaperContext()
    ins = pd.read_csv(sys.argv[1], header=None)
    fout = open('pdf.test.tsv', 'w')
    for v in ins.values:
        query = v[0]
        search.refresh_data(v[2])
        reslist = search.get_query_context(query, top_k=3)
        searchres = '||'.join(reslist).lower()
        aws = v[3].split('|')
        rel, sea, qas = 0, 0, 0
        # print(searchres, aws)
        for aw in aws:
            aw = aw.lower()
            rel += v[1].count(aw)
            sea += searchres.count(aw)
            # qas += qares.count(aw)
        print(query, [rel, sea], sea > qas,  sep='\t', file=fout)

if __name__ == "__main__":
    test_score()
    # get_lines()
    # ins = Paper()
    # ins.read_xml('./data/2022.findings-emnlp.146.pdf.tei.xml')
    # # ins.split_document()
    # for k in ins.section_lines:
    #     print(k)
    # print(ins.subindex)

    # ins.get_query_context('PLM排序是如何工作的？', 5)
    # while True:
    #     raw_text = input("\nContext prompt (stop to exit) >>> ")
    #     if not raw_text:
    #         print('Prompt should not be empty!')
    #         continue
    #     if raw_text == "stop":
    #         terminate_runs = 1
    #         break
    #     ins.get_query_context(raw_text, 5)
    # print(ins.paper.section_texts)
