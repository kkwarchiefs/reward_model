#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_generation.py
# @Author: 罗锦文
# @Date  : 2023/3/17
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime
import torch
import json

# path = "/search/ai/kaitongyang/online/model/GLM-10B-chinese-customization_03-07-21-23"
path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/glm_0.5"
# path = '/search/ai/pretrain_models/glm-large-chinese/'
device = "cuda:" + sys.argv[2]
suffix = "[回答][gMASK]"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
model = model.half().to(device)
model.eval()

def build_inputs(texts):
    inputs_ori = tokenizer(texts+suffix, padding=True, return_tensors="pt")
    inputs_ori = inputs_ori.to(device)
    for key in inputs_ori:
        inputs_ori[key] = inputs_ori[key][:, :-1]
    inputs = tokenizer.build_inputs_for_generation(inputs_ori, max_gen_length=256)
    return inputs

def generation_template(inputs, max_length, top_k, top_p, do_sample, temperature):
    outputs = model.generate(**inputs, max_length=max_length, eos_token_id=tokenizer.eop_token_id, top_k=top_k, top_p=top_p, do_sample=do_sample, temperature=temperature)
    response_text = [tokenizer.decode(logits) for logits in outputs[:, inputs["input_ids"].size()[1]:].tolist()]
    return response_text[0]



if __name__ == "__main__":
    for line in open(sys.argv[1]):
        try:
            prompt, resps = line.strip().split('\t')
        except:
            print(line, file=sys.stderr)
            continue
        response_list = json.loads(resps)
        inputs = build_inputs(prompt)
        try:
            print(prompt, generation_template(inputs, max_length=512, top_k=0, top_p=1.0, do_sample=True, temperature=1), sep='\t')
        except:
            continue
        # for candidate in response_list:
        #     if candidate['level'] != 1:
        #         continue
        #     print(prompt, (candidate['name'].replace('\t', ''),), candidate['level'], sep='\t')
        #     break
        # print(prompt, generation_template(inputs, max_length=256, top_k=20, top_p=0.6, do_sample=False, temperature=1.))
        # print(prompt, generation_template(inputs, max_length=128, top_k=0, top_p=1.0, do_sample=True, temperature=0.9), 0.9, sep='\t')
        # print(prompt, generation_template(inputs, max_length=128, top_k=0, top_p=1.0, do_sample=True, temperature=0.7), 0.7, sep='\t')
        # print(prompt, generation_template(inputs, max_length=128, top_k=0, top_p=1.0, do_sample=True, temperature=0.5), 0.5, sep='\t')
        # print(prompt, generation_template(inputs, max_length=128, top_k=0, top_p=1.0, do_sample=True, temperature=0.3), 0.3, sep='\t')
