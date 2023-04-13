#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : create_countext.py
# @Author: 罗锦文
# @Date  : 2023/4/12
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import numpy as np
import faiss
import torch
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer
import langid
from typing import Optional, Tuple
import collections

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    idlist = [ex['id'] for ex in examples]
    example_id_to_index = {k: i for i, k in enumerate(idlist)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()


    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    if start_logits[start_index] < 0 or end_logits[end_index] < 0:
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.get("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.get("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]
    return all_nbest_json


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
        if a1[0] >= a2[0] and a1[1] <= a2[1]:
            if a2[2] > a1[2]:
                newa = (a2[0], a2[1], a2[2])
                self.dsu[a1] = newa
                self.dsu[a2] = newa
            else:
                self.dsu[a2] = a1
        if a1[0] <= a2[0] and a1[1] >= a2[1]:
            if a1[2] > a2[2]:
                newa = (a1[0], a1[1], a1[2])
                self.dsu[a1] = newa
                self.dsu[a2] = newa
            else:
                self.dsu[a1] = a2
        if (a1[0] <= a2[1] and a1[1] >= a2[1]) or (a1[0] <= a2[0] and a1[1] >= a2[0]):
            newa = (min(a1[0], a2[0]), max(a1[1], a2[1]), max(a1[2], a2[2]))
            self.dsu[a2] = newa
            self.dsu[a1] = newa
        return

class SearchContext():
    def __init__(self):
        self.model_name = "embedding_mul_onnx"  # 模型目录名/venus注册模型名称
        address = "10.212.207.33:8000"  # 机器地址
        self.triton_client = httpclient.InferenceServerClient(url=address)
        rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
        self.tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)

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
        inputs = self.cut_doc(text, piece_len=500)
        emb = self.get_embedding(inputs)
        self.doc_piece_emb_index = faiss.IndexFlatL2(emb.shape[1])
        self.doc_piece_emb_index.add(emb)
        return inputs

    def cut_doc(self, data, piece_len=750, single_piece_max_len=1700):
        if len(data)<single_piece_max_len:
            return [data]
        piece_data = []
        index = 0
        while index < len(data):
            piece_data.append(data[index: index + piece_len])
            index += piece_len
        return piece_data

    def get_query_context(self, query, top_k=3):
        query_emb = self.get_embedding(query)
        sim_list, index_list = self.doc_piece_emb_index.search(query_emb, min(top_k, len(self.doc_piece_list)))
        context = []
        index_list = index_list.tolist()[0]
        index_list.sort()
        for index in index_list:
            context.append(self.doc_piece_list[index])
        return context

class QAContext():
    def __init__(self):
        self.model_name = "QuestionAnswering_onnx"  # 模型目录名/venus注册模型名称
        address = "10.212.207.33:8000"  # 机器地址
        self.triton_client = httpclient.InferenceServerClient(url=address)
        rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
        self.tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
        self.break_punctations = {'！', '？', '!', '?', '。'}
        self.force_break_punctations = {',', ';', '，', '；'}
        self.MAX_SENT_LEN = 512
        self.MIN_SENT_LEN = 256
        self.BATCH_SIZE = 256
        self.max_seq_length = 384
        self.max_content_length = 1500

    def _break_sentence(self, text: str):
        # 转换成小写
        # text = text.lower()
        ret = list()
        # 长度较短，不进行分句
        if len(text) < self.MAX_SENT_LEN:
            ret.append(text)
            return ret
        # 按照标点分句
        last_pos = 0
        for i, c in enumerate(text):
            if c in self.break_punctations and i - last_pos > self.MIN_SENT_LEN:
                segment = text[last_pos:i]
                if len(segment) > self.MAX_SENT_LEN:
                    # 需要再次分句
                    subs = self._force_break_sentence(segment)
                    ret.extend(subs)
                else:
                    ret.append(segment)
                # 分隔符
                last_pos = i
        # 结尾仍有残余部分需要分句
        if last_pos < len(text):
            segment = text[last_pos:]
            if len(segment) > self.MAX_SENT_LEN:
                subs = self._force_break_sentence(segment)
                ret.extend(subs)
            else:
                ret.append(segment)
        return ret

    def _force_break_sentence(self, text: str):
        ret = list()

        # 已经满足要求
        if len(text) <= self.MAX_SENT_LEN:
            ret.append(text)
            return ret

        idx = self.MAX_SENT_LEN - 1
        while idx >= 0 and text[idx] not in self.force_break_punctations:
            idx -= 1

        break_idx = idx + 1 if idx >= 0 else self.MAX_SENT_LEN
        ret.append(text[:break_idx])

        # 剩余部分
        remains = text[break_idx:]
        ret.extend(self._force_break_sentence(remains))
        return ret

    def _cal_context_idx(self, context_parts):
        index_parts = [0]
        start = 0
        for txt in context_parts[:-1]:
            start += len(txt)
            index_parts.append(start)
        return index_parts

    def _get_embedding(self, RM_batch):
        inputs = []
        inputs.append(httpclient.InferInput('input_ids', list(RM_batch[0].shape), 'INT64'))
        inputs.append(httpclient.InferInput('attention_mask', list(RM_batch[1].shape), 'INT64'))
        inputs[0].set_data_from_numpy(RM_batch[0])
        inputs[1].set_data_from_numpy(RM_batch[1])
        start_logits = httpclient.InferRequestedOutput('start_logits')
        end_logits = httpclient.InferRequestedOutput('end_logits')
        # try:
        results = self.triton_client.infer(
            self.model_name,
            inputs,
            model_version='1',
            outputs=[start_logits, end_logits],
            request_id='1'
        )
        # results = results.as_numpy('output')
        return results.as_numpy('start_logits'), results.as_numpy('end_logits')

    def _prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        sample_mapping = None
        if self.is_chinese:
            tokenized_examples = self.tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=self.max_seq_length,
                # stride=128,
                # return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=True
            )
        else:
            tokenized_examples = self.tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=self.max_seq_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=True
            )
            # print(len(tokenized_examples["input_ids"]))
            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # print(tokenizer.decode(tokenized_examples["input_ids"][i]))
            # print(tokenized_examples["input_ids"][i])

            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1
            if self.is_chinese:
                tokenized_examples["example_id"].append(examples["id"][i])
            else:
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def _process(self, example_batch):
        tokenized_examples = self._prepare_validation_features(example_batch)
        input_ids, attention_mask = np.array(tokenized_examples['input_ids']), np.array(tokenized_examples['attention_mask'])  # 模型测试输入数据
        steps = input_ids.shape[0] // self.BATCH_SIZE
        start_logits, end_logits = [], []
        for i in range(steps+1):
            RM_batch = (input_ids[i*self.BATCH_SIZE: (i+1)*self.BATCH_SIZE, :], attention_mask[i*self.BATCH_SIZE: (i+1)*self.BATCH_SIZE, :])
            tups = self._get_embedding(RM_batch)
            start_logits.append(tups[0])
            end_logits.append(tups[1])
        predict_tuple = (np.concatenate(start_logits), np.concatenate(end_logits))
        feautures = []
        for i in range(len(tokenized_examples['example_id'])):
            feautures.append({k: tokenized_examples[k][i] for k in tokenized_examples.keys()})
        examples = []
        for i in range(len(example_batch['id'])):
            examples.append({k: example_batch[k][i] for k in example_batch.keys()})
        all_nbest_json = postprocess_qa_predictions(examples, feautures, predict_tuple)
        return all_nbest_json

    def _merge_set(self, spans):
        ins = DSU()
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                ins.union(spans[i], spans[j])
        spanset = list(set([ins.find(a) for a in spans]))
        spanset.sort(key=lambda x: x[2], reverse=True)
        return spanset

    def _find_left(self, left, spanlen):
        if left < spanlen:
            return 0
        else:
            leftstar = left - spanlen
            while self.content[leftstar] not in self.break_punctations:
                leftstar += 1
                if leftstar == left:
                    break
            if leftstar == left:
                leftstar = left - spanlen
                while self.content[leftstar] not in self.force_break_punctations:
                    leftstar += 1
                    if leftstar == left:
                        break
            if leftstar == left:
                return left - spanlen
            else:
                return leftstar + 1

    def _find_right(self, right, spanlen):
        if right + spanlen >= len(self.content):
            return len(self.content)
        else:
            rightend = right + spanlen
            while self.content[rightend] not in self.break_punctations:
                rightend -= 1
                if rightend == right:
                    break
            if rightend == right:
                rightend = right + spanlen
                while self.content[rightend] not in self.force_break_punctations:
                    rightend -= 1
                    if rightend == right:
                        break
            if rightend == right:
                return right + spanlen
            else:
                return rightend

    def _find_parts(self, spanset, partlen):
        newspan = []
        for a in spanset:
            if partlen < 128:
                left_text = self._find_left(a[0], 64)
                right_text = self._find_right(a[1], 64)
                newspan.append((left_text, right_text, a[2]))
                break
            left_text = self._find_left(a[0], partlen // 4)
            right_text = self._find_right(a[1], partlen // 4)
            partlen = partlen + left_text - right_text
            newspan.append((left_text, right_text, a[2]))
        newspan = self._merge_set(newspan)
        return newspan

    @staticmethod
    def find_left_idx(char2idx, idx2pair, leftid, spanlen):
        left = char2idx[leftid]
        if left < spanlen:
            return idx2pair[0][0], 0
        else:
            return idx2pair[left - spanlen][0], left - spanlen
    @staticmethod
    def find_right_idx(char2idx, idx2pair, righttid, spanlen):
        right = char2idx[righttid]
        if right + spanlen > len(idx2pair):
            return idx2pair[len(idx2pair) - 1][1], len(idx2pair) - 1
        else:
            return idx2pair[right + spanlen][1], right + spanlen

    def _find_parts_idx(self, char2idx, idx2pair, spanset, partlen):
        newspan = []
        for a in spanset:
            if partlen < 128:
                text_left, token_left = QAContext.find_left_idx(char2idx, idx2pair, a[0], 128)
                text_right, token_right = QAContext.find_right_idx(char2idx, idx2pair, a[1], 128)
                newspan.append((text_left, text_right, a[2]))
                break
            text_left, token_left = QAContext.find_left_idx(char2idx, idx2pair, a[0], partlen // 4)
            text_right, token_right = QAContext.find_right_idx(char2idx, idx2pair, a[1], partlen // 4)
            partlen = partlen + token_left - token_right
            newspan.append((text_left, text_right, a[2]))
        newspan = self._merge_set(newspan)
        return newspan

    def _prepare_inputs(self, query):
        if self.is_chinese:
            example_batch = {
                "id": [str(i) for i in range(len(self.context_inputs))],
                "question": [query] * len(self.context_inputs),
                "context": self.context_inputs,
                "answers": [{"text": [], "answer_start": []}] * len(self.context_inputs),
            }
        else:
            example_batch = {
                "id": ['0'],
                "question": [query],
                "context": [self.content],
                "answers": [{"text": [], "answer_start": []}],
            }

        all_nbest_json = self._process(example_batch)
        all_span = []
        for k, vlist in all_nbest_json.items():
            start = 0
            if self.is_chinese:
                start = self.index_parts[int(k)]
            spans = [(a['offsets'][0] + start, a['offsets'][1] + start, a['score']) for a in vlist if 'offsets' in a]
            all_span += spans
        spanset = self._merge_set(all_span)
        if self.is_chinese:
            return self._find_parts(spanset, self.max_content_length)
        tokenized_offsets = self.tokenizer(
            self.content,
            truncation=True,
            max_length=len( self.content),
            return_offsets_mapping=True,
        )
        offset_mapping = tokenized_offsets["offset_mapping"]
        charid2idx = {}
        idx2pair = {}
        last_idx = 0
        for idx, (start, end) in enumerate(offset_mapping):
            if start != last_idx:
                for i in range(last_idx, start):
                    charid2idx[i] = idx
            last_idx = end
            for i in range(start, end):
                charid2idx[i] = idx
            idx2pair[idx] = (start, end)
        return self._find_parts_idx(charid2idx, idx2pair, spanset, self.max_content_length)

    def refresh_data(self, text):
        self.content = text.replace("\n", '')
        res = langid.classify(self.content)
        self.is_chinese = False
        if res[0] == 'zh':
            self.is_chinese = True
            context_parts = self._break_sentence(self.content)
            self.index_parts = self._cal_context_idx(context_parts)
            self.context_inputs = []
            for i in range(len(context_parts)):
                if i != len(context_parts) - 1:
                    self.context_inputs.append(context_parts[i] + context_parts[i + 1])
                else:
                    self.context_inputs.append(context_parts[i])

    def get_query_context(self, query, top_k=3):
        parts = self._prepare_inputs(query)
        # print(parts)
        res = [self.content[a[0]:a[1]] for a in parts]
        # print(res)
        if len(parts)  <= top_k:
            # res.reverse()
            return res
        else:
            tail = res[top_k:]
            # tail.reverse()
            tailstr = ' '.join(tail)
            newres = res[:top_k] + [tailstr]
            # newres.reverse()
            return newres

if __name__ == "__main__":
    ins = QAContext()
    text = open(sys.argv[1]).read()
    query = sys.argv[2]
    ins.refresh_data(text)
    reslist = ins.get_query_context(query)
    print(reslist)
    print(len(''.join(reslist)))
