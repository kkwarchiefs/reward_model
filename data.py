#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data.py
# @Author: 罗锦文
# @Date  : 2023/2/24
# @Desc  : 
import json
import random
from dataclasses import dataclass
from collections import defaultdict
import datasets
from typing import Union, List, Tuple, Dict
import codecs
import torch
from torch.utils.data import Dataset
from arguments import DataTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import pandas as pd
@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)

class TrainDatasetTask(Dataset):
    def __init__(
            self,
            args: DataTrainingArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)


    def __len__(self):
        return self.total_len

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_seq_length,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]['text'].split('\t')
        item = self.create_one_example(group[0])
        item['label'] = int(group[1])
        return item

class GroupedTrainDataset(Dataset):
    def __init__(
            self,
            args: DataTrainingArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_tsv,
        )['train']
        self.args = args
        self.tokenizer = tokenizer
        self.total_len = len(self.nlp_dataset)

    def create_one_example(self, prompt_inputs, resp, cur_max_length):
        inputs = self.tokenizer.build_inputs_for_generation(prompt_inputs, targets=resp,
                                                                 max_gen_length=cur_max_length,
                                                                 padding=True)
        inputs_idx = inputs['input_ids'].tolist()[0]
        start_idx = inputs_idx.index(50006)
        try:
            end_idx = inputs_idx.index(50007) - 1
        except:
            end_idx = inputs['input_ids'].shape[1] - 1
        # print(end_idx, inputs['input_ids'].shape)
        generation_mask = [0] * start_idx + [1] * (end_idx - start_idx) + [0] * (
                    inputs['input_ids'].shape[1] - end_idx)
        inputs['generation_mask'] = torch.tensor([generation_mask])
        return inputs

    def __len__(self):
        return self.total_len


    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.nlp_dataset[item]['text'].split('\t')
        prompt = group[0] + "[gMASK]"
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        response_list = json.loads(group[1])
        response_dict = defaultdict(list)
        for candidate in response_list:
            response_dict[int(candidate['level'])].append(candidate['name'])
        response_dict = sorted(response_dict.items(), key=lambda kv: kv[0], reverse=True)
        temp_response_pair = []
        for i in range(len(response_dict)):
            rsp_lefts = response_dict[i][1]
            for left in rsp_lefts:
                for j in range(i+1, len(response_dict)):
                    rsp_rights = response_dict[j][1]
                    for right in rsp_rights:
                        temp_response_pair.append([left, right])
        if len(temp_response_pair) == 0:
            return self.__getitem__(random.randint(0, self.total_len))
        group_batch = []
        examples = []
        if len(temp_response_pair) < self.args.train_group_size:
            negs = random.choices(temp_response_pair, k=self.args.train_group_size)
        else:
            negs = random.sample(temp_response_pair, k=self.args.train_group_size)
        cur_max_length = 512 - prompt_inputs['input_ids'].shape[1]
        for neg_entry in negs:
            left_inputs = self.create_one_example(prompt_inputs, neg_entry[0], cur_max_length)
            right_inputs = self.create_one_example(prompt_inputs, neg_entry[1], cur_max_length)
            examples.append({k: v.squeeze(0) for k, v in left_inputs.items()})
            examples.append({k: v.squeeze(0) for k, v in right_inputs.items()})

        # for e in examples:
        #     group_batch.append(self.create_one_example(e))
        return examples
