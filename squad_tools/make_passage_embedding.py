#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : make_passage_embedding.py
# @Author: 罗锦文
# @Date  : 2023/4/28
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import torch
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer
import json
class EmbeddingClent():
    def __init__(self):
        # self.model_name = "embedding_mul_onnx"  # 模型目录名/venus注册模型名称
        self.model_name = "embedding_pooling_onnx"
        address = "10.212.207.33:8000"  # 机器地址
        self.triton_client = httpclient.InferenceServerClient(url=address)
        rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
        self.tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)


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
        results = None
        try:
            results = self.triton_client.infer(
                self.model_name,
                inputs,
                model_version='1',
                outputs=[output],
                request_id='1'
            )
            results = results.as_numpy('output')
        except:
            pass
        return results
def du_embed():
    fout = open(sys.argv[1], 'wb')
    input_json = []
    client = EmbeddingClent()
    idx2embed = {}
    import time
    import pickle
    now = time.time()
    for line in sys.stdin:
        input_json.append(json.loads(line))
        if len(input_json) == 200:
            ids, texts = [], []
            for a in input_json:
                ids.append(a['paragraph_id'])
                texts.append(a['paragraph_text'])
            results = client.get_embedding(texts)
            if results is not None:
                for x, y in zip(ids, results):
                    idx2embed[x] = y
            input_json = []
    print(len(idx2embed))
    pickle.dump(idx2embed, fout)
    fout.close()

def read_json():
    fout = open(sys.argv[1], 'wb')
    input_json = []
    client = EmbeddingClent()
    idx2embed = {}
    import time
    import pickle
    now = time.time()
    for line in sys.stdin:
        input_json.append(line)
        if len(input_json) == 200:
            ids, texts = [], []
            for a in input_json:
                items = a.strip().split('\t')
                if len(items) != 2:
                    continue
                ids.append(items[0])
                texts.append(items[1])
            results = client.get_embedding(texts)
            if results is not None:
                for x, y in zip(ids, results):
                    idx2embed[x] = y
            input_json = []
    print(len(idx2embed))
    pickle.dump(idx2embed, fout)
    fout.close()

if __name__ == "__main__":
    fout = open(sys.argv[1], 'wb')
    input_json = []
    client = EmbeddingClent()
    idx2embed = {}
    import time
    import pickle
    now = time.time()
    for line in sys.stdin:
        input_json.append(line)
        if len(input_json) == 200:
            ids, texts = [], []
            for a in input_json:
                items = a.strip().split('\t')
                if len(items) != 3:
                    continue
                ids.append(items[0] + '_' + items[1])
                texts.append(items[1])
            results = client.get_embedding(texts)
            if results is not None:
                for x, y in zip(ids, results):
                    idx2embed[x] = y
            input_json = []
    print(len(idx2embed))
    pickle.dump(idx2embed, fout)
    fout.close()
