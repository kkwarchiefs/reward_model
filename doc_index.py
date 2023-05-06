#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : doc_index.py
# @Author: 罗锦文
# @Date  : 2023/5/5
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
from create_countext import DSU
import numpy as np
model_path = "/search/ai/pvopliu/glm_10m/GLM/GLM/convert_scripts/glm_10b_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

class RerankContext():
    def __init__(self):
        # self.model_name = "embedding_mul_onnx"  # 模型目录名/venus注册模型名称
        self.model_name = "colbert_mul_onnx"
        self.rerank_name = "rerank_mul_onnx"
        address = "10.164.164.172:89999"  # 机器地址
        self.triton_client = httpclient.InferenceServerClient(url=address)
        rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
        self.tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
        self.piece_len = 330

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


    def get_rerank_score(self, query, doc):
        RM_input = self.tokenizer('Q:' + query + 'A:' + doc, max_length=512, truncation=True, return_tensors="pt", padding=True)
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
            self.rerank_name,
            inputs,
            model_version='1',
            outputs=[output],
            request_id='1'
        )
        results = results.as_numpy('output')
        return results

    def get_doc_embedding_index(self, text):
        # inputs, self.offsets, self.fulltext = RerankContext.cut_doc_move(text, piece_len=self.piece_len)
        inputs = RerankContext.cut_doc_plus(text, piece_len=self.piece_len)
        self.doc_embedding = self.get_embedding(inputs[:256])

        return inputs

    @staticmethod
    def cut_doc_plus(data, piece_len=400):
        tokens = tokenizer(data)
        tokens_ids = tokens['input_ids'][1:-2]
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
        print(piece_data)
        return piece_data

    @staticmethod
    def cut_doc_move(data, piece_len=400):
        tokens = tokenizer(data)
        tokens_ids = tokens['input_ids'][1:-2]
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
        print('='*20)
        print(query)
        context = []
        new_score = []
        for index, para in enumerate(self.doc_piece_list):
            rank_score = self.get_rerank_score(query, para)[0]
            new_score.append((rank_score[1], self.doc_piece_list[index].replace("\n", "<n>"), index, rank_score))
        new_score.sort(key=lambda a: a[0], reverse=True)
        for data in new_score[:top_k]:
            print(data)
            context.append(data[1])
        return context

    def get_query_context_colbert(self, query, top_k=3):
        query_emb = self.get_embedding(query)
        # scores = np.matmul(
        #     query_emb,
        #     self.doc_embedding.transpose(1, 0))[0]

        scores = (query_emb @ self.doc_embedding.transpose(0, 2, 1)).max(2).sum(1)
        context = []
        index_list = np.argsort(scores)[::-1]
        print('='*20)
        print(query)
        for index in index_list:
            doc_text = self.doc_piece_list[index]
            context.append(doc_text)
            # rank_score = self.get_rerank_score(query, doc_text)[0]
            print(self.doc_piece_list[index].replace("\n", "<n>"), index, scores[index])
            if len(context) == top_k:
                break
        return context

    def get_query_context_move(self, query, top_k=3):
        print('='*20)
        print(query)
        new_score = []
        for index, para in enumerate(self.doc_piece_list):
            context.append(self.doc_piece_list[index])
            context_span.append(self.offsets[index])
            rank_score = self.get_rerank_score(query, para)[0]
            new_score.append((rank_score[1], self.doc_piece_list[index].replace("\n", "<n>"), index, scores[index], rank_score))
            # print(self.doc_piece_list[index].replace("\n", "<n>"), self.offsets[index], index, scores[index], rank_score, np.argmax(rank_score))
            # context_span = SearchContext._merge_set(context_span)
            # if sum([a[1] - a[0] for a in context_span]) >= 1000:
            #     break
        # print(context_span, sum([a[1] - a[0] for a in context_span]))
        new_score.sort(key=lambda a:a[0], reverse=True)
        for data in new_score:
            print(data)
        context = [tokenizer.decode(self.fulltext[a[0]:a[1]]) for a in context_span]
        # print(context)
        return context

