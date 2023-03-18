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
        print("self.total_len", self.total_len)

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
        return {k: v.squeeze(0) for k, v in inputs.items()}

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
            left_level = response_dict[i][0]
            for left in rsp_lefts:
                for j in range(i+1, len(response_dict)):
                    rsp_rights = response_dict[j][1]
                    right_level = response_dict[j][0]
                    if left_level - right_level <= 1:
                        continue
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
        cur_max_length = self.args.max_seq_length - prompt_inputs['input_ids'].shape[1]
        for neg_entry in negs:
            left_inputs = self.create_one_example(prompt_inputs, neg_entry[0], cur_max_length)
            right_inputs = self.create_one_example(prompt_inputs, neg_entry[1], cur_max_length)
            examples.append(left_inputs)
            examples.append(right_inputs)
        # assert len(examples) == self.args.train_group_size * 2
        # for e in examples:
        #     group_batch.append(self.create_one_example(e))
        return examples


class PredictionDataset(Dataset):

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_seq_length=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.nlp_dataset) - 1

    def create_one_example(self, prompt_inputs, prompt, resp, cur_max_length):
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
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __getitem__(self, item):
        group = self.nlp_dataset[item]['text'].split('\t')
        prompt = group[0] + "[gMASK]"
        resp = eval(group[1])[0]
        label = int(group[2])
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        cur_max_length = self.max_seq_length - prompt_inputs['input_ids'].shape[1]
        inputs = self.create_one_example(prompt_inputs, prompt, resp, cur_max_length)
        inputs['level_label'] = label
        return [inputs]

class GroupedTrainDatasetClassify(Dataset):
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

    # def create_one_example(self, prompt, resp):
    #     resp = resp.replace("\n", "<n>")
    #     inputs = self.tokenizer.encode_plus(
    #         prompt,
    #         resp,
    #         max_length=512,
    #         truncation=True,
    #         return_tensors="pt")
    #     # inputs.pop('token_type_ids')
    #     return {k: v.squeeze(0) for k, v in inputs.items()}
    def create_one_example(self, prompt, resp):
        resp = resp.replace("\n", "<n>")
        inputs = self.tokenizer(
            prompt + "[UNUSED1]" + resp,
            max_length=512,
            truncation=True,
            return_tensors="pt")
        inputs.pop('token_type_ids')
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __len__(self):
        return self.total_len


    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.nlp_dataset[item]['text'].split('\t')
        prompt = group[0]

        response_list = json.loads(group[1])
        response_dict = defaultdict(list)
        for candidate in response_list:
            response_dict[int(candidate['level'])].append(candidate['name'])
        response_dict = sorted(response_dict.items(), key=lambda kv: kv[0], reverse=True)
        temp_response_pair = []
        for i in range(len(response_dict)):
            rsp_lefts = response_dict[i][1]
            left_level = response_dict[i][0]
            for left in rsp_lefts:
                for j in range(i+1, len(response_dict)):
                    rsp_rights = response_dict[j][1]
                    right_level = response_dict[j][0]
                    if left_level - right_level <= 1:
                        continue
                    for right in rsp_rights:
                        temp_response_pair.append([left, right])
        if len(temp_response_pair) == 0:
            return self.__getitem__(random.randint(0, self.total_len))
        examples = []
        if len(temp_response_pair) < self.args.train_group_size:
            negs = random.choices(temp_response_pair, k=self.args.train_group_size)
        else:
            negs = random.sample(temp_response_pair, k=self.args.train_group_size)
        for neg_entry in negs:
            left_inputs = self.create_one_example(prompt, neg_entry[0])
            right_inputs = self.create_one_example(prompt, neg_entry[1])
            examples.append(left_inputs)
            examples.append(right_inputs)
        # assert len(examples) == self.args.train_group_size * 2
        # for e in examples:
        #     group_batch.append(self.create_one_example(e))
        return examples

class PredictionDatasetClassify(Dataset):

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_seq_length=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.nlp_dataset)

    # def create_one_example(self, prompt, resp):
    #     resp = resp.replace("\n", "<n>")
    #     inputs = self.tokenizer.encode_plus(
    #         prompt,
    #         resp,
    #         max_length=512,
    #         truncation=True,
    #         return_tensors="pt")
    #     # inputs.pop('token_type_ids')
    #     return {k: v.squeeze(0) for k, v in inputs.items()}
    def create_one_example(self, prompt, resp):
        resp = resp.replace("\n", "<n>")
        inputs = self.tokenizer(
            prompt + "[UNUSED1]" + resp,
            max_length=512,
            truncation=True,
            return_tensors="pt")
        inputs.pop('token_type_ids')
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __getitem__(self, item):
        group = self.nlp_dataset[item]['text'].split('\t')
        prompt = group[0]
        resp = eval(group[1])[0]
        label = int(group[2])
        inputs = self.create_one_example(prompt, resp)
        inputs['level_label'] = label
        return [inputs]

class GroupedTrainDatasetClassifyList(Dataset):
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
        self.total_len = len(self.nlp_dataset) - 1

    # def create_one_example(self, prompt, resp):
    #     resp = resp.replace("\n", "<n>")
    #     inputs = self.tokenizer.encode_plus(
    #         prompt,
    #         resp,
    #         max_length=512,
    #         truncation=True,
    #         return_tensors="pt")
    #     # inputs.pop('token_type_ids')
    #     return {k: v.squeeze(0) for k, v in inputs.items()}
    def create_one_example(self, prompt, resp):
        resp = resp.replace("\n", "<n>")
        inputs = self.tokenizer(
            prompt + "[UNUSED1]" + resp,
            max_length=512,
            truncation=True,
            return_tensors="pt")
        inputs.pop('token_type_ids')
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.nlp_dataset[item]['text'].split('\t')
        prompt = group[0]

        response_list = json.loads(group[1])
        response_dict = defaultdict(list)
        for candidate in response_list:
            response_dict[int(candidate['level'])].append(candidate['name'])

        response_dict = sorted(response_dict.items(), key=lambda kv: kv[0], reverse=True)
        temp_response_pair = []
        for i in range(len(response_dict)):
            rsp_lefts = response_dict[i][1]
            left_level = response_dict[i][0]
            for left in rsp_lefts:
                right_all = []
                for j in range(i+1, len(response_dict)):
                    rsp_rights = response_dict[j][1]
                    right_level = response_dict[j][0]
                    if left_level - right_level <= 1:
                        continue
                    for right in rsp_rights:
                        right_all.append(right)
                if len(right_all) < self.args.rank_list_size + 2:
                    continue
                else:
                    right_negs = random.sample(right_all, k=self.args.rank_list_size)
                temp_response_pair.append([left] + right_negs)
        if len(temp_response_pair) == 0:
            return self.__getitem__(random.randint(0, self.total_len))
        examples = []
        if len(temp_response_pair) < self.args.train_group_size:
            negs = random.choices(temp_response_pair, k=self.args.train_group_size)
        else:
            negs = random.sample(temp_response_pair, k=self.args.train_group_size)
        for neg_entry in negs:
            for left_entry in neg_entry:
                left_inputs = self.create_one_example(prompt, left_entry)
                examples.append(left_inputs)
        # assert len(examples) == self.args.train_group_size * 2
        # for e in examples:
        #     group_batch.append(self.create_one_example(e))
        return examples


class GroupedTrainDatasetClassifyRandom(Dataset):
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
        self.total_len = len(self.nlp_dataset) - 1
        self.prompt2gpt = self.read_file('/search/ai/kaitongyang/RLHF_DEBUG/RM/data/chatgpt_we/prompt.txt')

    def read_file(self, file_name):
        prompt2gpt = {}
        for line in open(file_name):
            ins = json.loads(line)
            prompt2gpt[ins["prompt"]] = ins["best"]
        return prompt2gpt

    def create_one_example(self, prompt, resp):
        resp = resp.replace("\n", "<n>")
        inputs = self.tokenizer(
            prompt + "[UNUSED1]" + resp,
            max_length=512,
            truncation=True,
            return_tensors="pt")
        inputs.pop('token_type_ids')
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __len__(self):
        return self.total_len

    def add_some_unmask(self):
        words = [random.randint(100, 40000) for _ in range(random.randint(10, 30))]
        resp = self.tokenizer.decode(words)
        return resp

    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.nlp_dataset[item]['text'].split('\t')
        prompt = group[0]

        response_list = json.loads(group[1])
        response_dict = defaultdict(list)
        res_all = []
        for candidate in response_list:
            ends = random.randint(20, 60)
            res_all.append(candidate['name'])
            response_dict[int(candidate['level'])].append(candidate['name'][:ends])
        response_dict[0].append(self.add_some_unmask())
        for rsp_ in  random.sample(res_all, k=3):
            response_dict[0].append(rsp_[:20]+self.add_some_unmask())
        chat_resp = self.prompt2gpt.get(prompt)
        if chat_resp is not None:
            ends = random.randint(20, 60)
            response_dict[10].append(chat_resp[:ends])
        response_dict = sorted(response_dict.items(), key=lambda kv: kv[0], reverse=True)
        temp_response_pair = []
        for i in range(len(response_dict)):
            rsp_lefts = response_dict[i][1]
            left_level = response_dict[i][0]
            for left in rsp_lefts:
                right_all = []
                for j in range(i + 1, len(response_dict)):
                    rsp_rights = response_dict[j][1]
                    right_level = response_dict[j][0]
                    if left_level - right_level <= 1:
                        continue
                    for right in rsp_rights:
                        right_all.append(right)
                if len(right_all) < self.args.rank_list_size + 2:
                    continue
                else:
                    right_negs = random.sample(right_all, k=self.args.rank_list_size)
                temp_response_pair.append([left] + right_negs)
        if len(temp_response_pair) == 0:
            return self.__getitem__(random.randint(0, self.total_len))
        examples = []
        if len(temp_response_pair) < self.args.train_group_size:
            negs = random.choices(temp_response_pair, k=self.args.train_group_size)
        else:
            negs = random.sample(temp_response_pair, k=self.args.train_group_size)
        for neg_entry in negs:
            for left_entry in neg_entry:
                left_inputs = self.create_one_example(prompt, left_entry)
                examples.append(left_inputs)
        # assert len(examples) == self.args.train_group_size * 2
        # for e in examples:
        #     group_batch.append(self.create_one_example(e))
        return examples

class PredictionDatasetClassifyRandom(Dataset):

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_seq_length=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.nlp_dataset)

    # def create_one_example(self, prompt, resp):
    #     resp = resp.replace("\n", "<n>")
    #     inputs = self.tokenizer.encode_plus(
    #         prompt,
    #         resp,
    #         max_length=512,
    #         truncation=True,
    #         return_tensors="pt")
    #     # inputs.pop('token_type_ids')
    #     return {k: v.squeeze(0) for k, v in inputs.items()}
    def create_one_example(self, prompt, resp):
        resp = resp.replace("\n", "<n>")
        inputs = self.tokenizer(
            prompt + "[UNUSED1]" + resp,
            max_length=512,
            truncation=True,
            return_tensors="pt")
        inputs.pop('token_type_ids')
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __getitem__(self, item):
        group = self.nlp_dataset[item]['text'].split('\t')
        prompt = group[0]
        resp = eval(group[1])[0].replace("\n", "<n>")
        label = int(group[2])
        ends = random.randint(20, 60)
        inputs = self.create_one_example(prompt, resp[:ends])
        inputs['level_label'] = label
        return [inputs]
