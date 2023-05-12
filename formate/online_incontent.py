#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : online_incontent.py
# @Author: 罗锦文
# @Date  : 2023/5/8
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import torch
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer
import langid
from typing import Optional, Tuple
import collections
import json
from  bm25_search import  *
model_path = "/search/ai/pvopliu/glm_10m/GLM/GLM/convert_scripts/glm_10b_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

class DSU:
    def __init__(self):
        self.dsu = {}

    def find(self, account):
        if account not in self.dsu:
            self.dsu[account] = account
            return account
        if account == self.dsu[account]:
            return account
        self.dsu[account] = self.find(self.dsu[account])
        return self.dsu[account]

    def union(self, x, y):
        a1 = self.find(x)
        a2 = self.find(y)

        if (a1[0] <= a2[1] and a1[1] >= a2[1]) or (a1[0] <= a2[0] and a1[1] >= a2[0]):
            newa = (min(a1[0], a2[0]), max(a1[1], a2[1]), max(a1[2], a2[2]))
            self.dsu[a2] = newa
            self.dsu[a1] = newa
        return
    def union_pos(self, x, y):
        a1 = self.find(x)
        a2 = self.find(y)
        if (a1[0] <= a2[1] and a1[1] >= a2[1]) or (a1[0] <= a2[0] and a1[1] >= a2[0]):
            newa = (min(a1[0], a2[0]), max(a1[1], a2[1]))
            self.dsu[a2] = newa
            self.dsu[a1] = newa
        return


class ReaderContext():
    def __init__(self):
        self.model_name = "embedding_mul_onnx_v2"  # 模型目录名/venus注册模型名称
        # self.model_name = "embedding_pooling_onnx"
        self.rerank_name = "rerank_mul_onnx"
        address = "10.164.164.172:89999"  # 机器地址
        self.triton_client = httpclient.InferenceServerClient(url=address)
        rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
        self.tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
        self.piece_len = 334

    def refresh_data(self, text):
        self.doc_piece_list = self.get_doc_embedding_index(text)

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

    def get_doc_embedding_index(self, text):
        # inputs, self.offsets, self.fulltext = ReaderContext.cut_doc_move(text, piece_len=self.piece_len)
        inputs = ReaderContext.cut_doc_plus(text, piece_len=self.piece_len)
        # print(inputs, offsets)
        self.doc_embedding = self.get_embedding(inputs[:12])
        return inputs


    @staticmethod
    def cut_doc_plus(data, piece_len=400, single_piece_max_len=1000):
        tokens = tokenizer(data)
        tokens_ids = tokens['input_ids'][1:-2]
        # if len(tokens_ids) < single_piece_max_len:
        #     return [data]
        index = 0
        piece_data = []
        last_index = 0
        while index < len(tokens_ids):
            index += piece_len
            if index < len(tokens_ids):
                temp_data = tokenizer.decode(tokens_ids[last_index: index])
            else:
                temp_data = tokenizer.decode(tokens_ids[last_index:])
            piece_data.append(temp_data)
            last_index = index
        return piece_data

    @staticmethod
    def cut_doc_move(data, piece_len=400, single_piece_max_len=1000):
        tokens = tokenizer(data)
        tokens_ids = tokens['input_ids'][1:-2]
        if len(tokens_ids) < single_piece_max_len:
            return [data], [(0, len(tokens_ids))], tokens_ids
        index = 0
        piece_data = []
        piece_half = piece_len // 2
        piece_length = []
        # fulltext = tokenizer.decode(tokens_ids)
        while index < len(tokens_ids):
            if index == 0:
                index += piece_len
            else:
                index += piece_half
            # last_index = max(index-piece_len, 0)
            if index < len(tokens_ids):
                temp_data = tokenizer.decode(tokens_ids[index-piece_len: index])
                piece_length.append((index-piece_len, index))
            else:
                temp_data = tokenizer.decode(tokens_ids[index-piece_len:])
                piece_length.append((index - piece_len, len(tokens_ids)))
            piece_data.append(temp_data)
            # piece_length.append((start, start+len(temp_data)))
            # start += len(tokenizer.decode(tokens_ids[index-piece_len: index-piece_half]))
        return piece_data, piece_length, tokens_ids


    @staticmethod
    def _merge_set(spans):
        ins = DSU()
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                ins.union_pos(spans[i], spans[j])
        spanset = list(set([ins.find(a) for a in spans]))
        spanset.sort(key=lambda x: x[0])
        return spanset

    def get_query_context(self, query, top_k=3):
        query_emb = self.get_embedding(query)
        scores = np.matmul(
            query_emb,
            self.doc_embedding.transpose(1, 0))[0]
        context = []
        index_list = np.argsort(scores)[::-1]
        print('='*20)
        print(query)
        for index in index_list:
            context.append(self.doc_piece_list[index])
            print(self.doc_piece_list[index].replace("\n", "<n>"), index, scores[index])
            if len(context) == top_k:
                break
        return context

    def get_query_context_move(self, query, top_k=3):
        if len(self.doc_piece_list) == 1:
            return self.doc_piece_list
        query_emb = self.get_embedding(query)
        scores = np.matmul(
            query_emb,
            self.doc_embedding.transpose(1, 0))[0]
        context = []
        context_span = []
        index_list = np.argsort(scores)[::-1]

        for index in index_list:
            context.append(self.doc_piece_list[index])
            context_span.append(self.offsets[index])
            context_span = ReaderContext._merge_set(context_span)
            if sum([a[1] - a[0] for a in context_span]) >= 1000:
                break
        # context = [tokenizer.decode(self.fulltext[a[0]:a[1]]) for a in context_span]
        context_span_split = []
        # 小于3段的拆分成三端
        if len(context_span) == 1:
            part_length = (context_span[0][1] - context_span[0][0]) // 3
            start_index = context_span[0][0]
            end_index = context_span[0][1]
            context_span_split = [(start_index, start_index + part_length),
                                  (start_index + part_length, start_index + 2*part_length),
                                  (start_index + 2*part_length, end_index)]
        elif len(context_span) == 2:
            context_span_split.append(context_span[0])
            part_length = (context_span[1][1] - context_span[1][0]) // 2
            start_index = context_span[1][0]
            end_index = context_span[1][1]
            context_span_split += [(start_index, start_index + part_length),
                                   (start_index + part_length, end_index)]
        else:
            context_span_split = context_span

        context_chunks = [tokenizer.decode(self.fulltext[a[0]:a[1]]) for a in context_span_split]
        # 大于三段的合并成三段
        if len(context_chunks) > 3:
            context_chunks = context_chunks[:2] + [''.join(context_chunks[2:])]
        return context_chunks


if __name__ == "__main__":
    pass
