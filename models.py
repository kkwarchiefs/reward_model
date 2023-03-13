#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : models.py
# @Author: 罗锦文
# @Date  : 2023/3/11
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    ModelOutput,
)
from torch import nn
from arguments import *
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

class Reranker(nn.Module):
    def __init__(self, hf_model, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.data_args = data_args
        self.train_args = train_args
        self.project = nn.Linear(hf_model.config.hidden_size, 1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size * self.data_args.train_group_size,
                        dtype=torch.long)
        )


    def forward(self, **batch):
        # print("batch", batch)
        # print(batch['input_ids'].device, self.hf_model.device)
        # print(batch.keys())
        batch['input_ids'] = batch['input_ids'].to(self.hf_model.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.hf_model.device)
        if 'position_ids' in batch:
            batch['position_ids'] = batch['position_ids'].to(self.hf_model.device)
        if 'generation_mask' in batch:
            batch['generation_mask'] = batch['generation_mask'].to(self.hf_model.device)
        # batch = batch.to(self.accelerator.device)
        ranker_out = self.hf_model(**batch, return_dict=True)
        last_hidden_states = ranker_out.last_hidden_states * batch['generation_mask'].unsqueeze(2)
        pooling_hidden_states = torch.sum(last_hidden_states, axis=1) / torch.sum(batch['generation_mask'], axis=1).unsqueeze(1)
        logits = torch.sigmoid(self.project(pooling_hidden_states))
        # Add a clip to confine scores in [0, 1].
        # logits = torch.clamp(logits, 0, 1)

        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size * self.data_args.train_group_size,
                2
            )
            loss = self.cross_entropy(scores, self.target_label)

            return ModelOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments, *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

