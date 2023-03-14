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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

class RerankerTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # print(self.eval_dataset, output.predictions)
        fout = open('predictions.txt', 'w')
        for ins, prediction in zip(self.eval_dataset, output.predictions):
            # print(ins)
            response = self.tokenizer.decode(ins[0]['input_ids'])
            response = response.replace("<|startofpiece|>", "").replace("<|endofpiece|>", "").replace("<|endoftext|>","")
            response = response.replace("<|startofpiece|>", "").replace("[回答]", "").replace("[CLS]", "").replace("\n", "").replace("<n>", "").replace(" ", "")
            print(response, prediction, ins[0]['level_label'], sep='\t', file=fout)
        fout.close()
        return output.metrics


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
        logits = self.project(pooling_hidden_states)
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
            return logits

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments, *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

