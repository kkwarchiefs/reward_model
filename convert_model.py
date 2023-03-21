#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : convert_model.py
# @Author: 罗锦文
# @Date  : 2023/3/14
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import torch
import os

def convert_model():
    RM_model_path = "./output/rm_model/"
    model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model.bin'), map_location="cpu")
    new_model_dict = {k.replace('hf_model.', ''): v for k, v in model_dict.items()}
    torch.save(new_model_dict, os.path.join(RM_model_path, 'model.pt'))

if __name__ == "__main__":
    RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/glm_0.5/"
    model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model.bin'), map_location="cpu")
    new_model_dict = {k: torch.zeros_like(v, dtype=v.dtype) for k, v in model_dict.items()}
    torch.save(new_model_dict, os.path.join("./", 'model.pt'))
