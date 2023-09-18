#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : span_match.py
# @Author: 罗锦文
# @Date  : 2023/9/8
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import numpy as np
from typing import Optional, Tuple
import tritonclient.http as httpclient
from transformers import AutoTokenizer
import logging
import collections
import os
import json
logger = logging.getLogger(__name__)
import pickle as pkl
import re
address = "10.164.163.210:89997"  # 机器地址
triton_client = httpclient.InferenceServerClient(url=address)

model_name = "TextReference_onnx"

RM_model_path = "/search/ai/pretrain_models/infoxlm-base/"

tokenizer = AutoTokenizer.from_pretrained(RM_model_path, trust_remote_code=True)
question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"
pad_on_right = tokenizer.padding_side == "right"
max_seq_length = 512

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
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

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            feature_show = []
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
                    # if start_logits[start_index] < 0 or end_logits[end_index] < 0:
                    #     continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
                    context = example["context"]
                    feature_show.append((offset_mapping[start_index][0], offset_mapping[end_index][1], context[offset_mapping[start_index][0]: offset_mapping[end_index][1]], start_logits[start_index], end_logits[end_index]))
            # print(feature_show)
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

def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
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
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def process(example_batch):
    tokenized_examples = prepare_validation_features(example_batch)
    RM_batch = (np.array(tokenized_examples['input_ids']), np.array(tokenized_examples['attention_mask']))  # 模型测试输入数据
    inputs = []
    inputs.append(httpclient.InferInput('input_ids', list(RM_batch[0].shape), 'INT64'))
    inputs.append(httpclient.InferInput('attention_mask', list(RM_batch[1].shape), 'INT64'))
    inputs[0].set_data_from_numpy(RM_batch[0])
    inputs[1].set_data_from_numpy(RM_batch[1])
    start_logits = httpclient.InferRequestedOutput('start_logits')
    end_logits = httpclient.InferRequestedOutput('end_logits')
    # try:
    results = triton_client.infer(
        model_name,
        inputs,
        model_version='1',
        outputs=[start_logits, end_logits],
        request_id='1'
    )
    # results = results.as_numpy('output')
    predict_tuple = (results.as_numpy('start_logits'), results.as_numpy('end_logits'))
    # print(predict_tuple[0][:, :5])
    all_nbest_json = None
    # print(tokenized_examples)
    feautures = []
    for i in range(len(tokenized_examples['example_id'])):
        feautures.append({k: tokenized_examples[k][i] for k in tokenized_examples.keys()})
    examples = []
    for i in range(len(example_batch['id'])):
        examples.append({k: example_batch[k][i] for k in example_batch.keys()})
    all_nbest_json = postprocess_qa_predictions(examples, feautures, predict_tuple)
    return all_nbest_json

def check_stop(start, end, stoplist):
    for num in stoplist:
        if start <= num <= end:
            return False
    return True
def process_all(query, content, stop_index):
    example_batch = {
        "id": ['0'],
        "question": [query],
        "context": [content],
        "answers": [{"text": [], "answer_start": []}],
    }
    all_nbest_json = process(example_batch)
    all_span = [(a['offsets'][0], a['offsets'][1], a['score']) for a in all_nbest_json['0'] if 'offsets' in a ]
    all_filter = [a for a in all_span if a[1] - a[0] > 5 and check_stop(a[0], a[1], stop_index)]
    all_return = [(content[a[0]:a[1]], a[2]) for a in all_filter]
    if len(all_filter):
        resp = content[all_filter[0][0]:all_filter[0][1]]
        return resp, all_filter[0], all_return
    else:
        resp = 'NULL'
    return resp, None, all_return
def __split_sentences(text):
    return re.split('[。？！\n；]', text)

def cover_index(index, start, end):
    start_idx = [i for i in index if i <= start]
    end_idx = [i for i in index if i >= end]
    res = [max(start_idx), min(end_idx)]
    # first = index.index(res[0])
    # second = index.index(res[1])
    return res[0], res[1]


def find_min_indices_of_max_value(dp):
    max_value = max(max(row) for row in dp)
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if dp[i][j] == max_value:
                return i, j
    return -1, -1  # 如果dp表为空或者没有找到最大值，返回-1, -1


def longest_common_subsequence(X, Y):
    m = len(X)
    n = len(Y)

    # 初始化DP表和direction表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    direction = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                direction[i + 1][j + 1] = 'diagonal'
            elif dp[i][j + 1] >= dp[i + 1][j]:  # 修改这里，优先考虑"up"方向
                dp[i + 1][j + 1] = dp[i][j + 1]
                direction[i + 1][j + 1] = 'up'
            else:
                dp[i + 1][j + 1] = dp[i + 1][j]
                direction[i + 1][j + 1] = 'left'

    # 追踪最长公共子序列
    i, j = find_min_indices_of_max_value(dp)
    span_X = [m, -1]
    span_Y = [n, -1]
    while i > 0 and j > 0:
        if direction[i][j] == 'diagonal':
            span_X = [min(span_X[0], i-1), max(span_X[1], i-1)]
            span_Y = [min(span_Y[0], j-1), max(span_Y[1], j-1)]
            i -= 1
            j -= 1
        elif direction[i][j] == 'up':
            i -= 1
        else:
            j -= 1

    return dp[m][n], X[span_X[0]:span_X[1]+1], Y[span_Y[0]:span_Y[1]+1]

def similarity(str1, str2):
    lcs_length, left, right = longest_common_subsequence(str1, str2)
    similarity = lcs_length / len(str1)
    return similarity, left, right
def read_datas():
    for line in sys.stdin:
        ins = json.loads(line)
        for a in ins['messages']:
            if a['role'] == 'user':
                context = a['content']
            if a['role'] == 'assistant':
                resp = a['content']
        index = context.find('\n根据上面内容回答问题：\n')
        if index == -1:
            continue
        query = context[index+len('\n根据上面内容回答问题：\n'):]
        context = context[:index]
        context = context.replace('其中第1部分文章内容是：', '\x01\x03').replace('其中第2部分文章内容是：', '\x01\x03')\
            .replace('其中第3部分文章内容是：', '\x01\x03').replace('其中第1部分内容是：', '\x01\x03').replace('其中第2部分内容是：', '\x01\x03').replace('其中第3部分内容是：', '\x01\x03')

        parts = context.split('\x01\x03')
        parts = [a for a in parts if len(a)]
        all_context = ''
        stop_index = [0]
        for a in parts:
            all_context += a
            stop_index.append(len(all_context))
        sents = __split_sentences(resp)
        sents = [a for a in sents if len(a) >= 5]
        res = []
        for sent in sents:
            before, _, _ = similarity(sent, all_context)
            if before < 0.8:
                continue
            span_text, idx, returns = process_all(sent, all_context, stop_index)
            # print(sent, span_text, idx, stop_index, sep= '\t')
            if idx and idx[2] < 2:
                index_tup = cover_index(stop_index, idx[0], idx[1])
                newtext = all_context[index_tup[0]:index_tup[1]]
                score, left, right = similarity(sent, newtext)
                print(sent, span_text.replace('\n', '<n>'), right.replace('\n', '<n>'), score, sep='\t')
                if score > 0.4:
                    res.append((sent, span_text, right, newtext, score))

def find_nearest_punctuation(s, fragment):
    punctuation = set('。？，！：”“')
    start = s.find(fragment)
    if start == -1:
        return None, None

    end = start + len(fragment) - 1
    left_punctuation = None
    right_punctuation = None

    # 查找最近的左侧标点符号
    for i in range(start - 1, -1, -1):
        if s[i] in punctuation:
            left_punctuation = i
            break

    # 查找最近的右侧标点符号
    for i in range(end + 1, len(s)):
        if s[i] in punctuation:
            right_punctuation = i
            break
    return left_punctuation, right_punctuation

import hashlib

def get_md5(sign):
    instance = hashlib.md5()
    instance.update(sign.encode("utf-8"))
    return instance.hexdigest()
def lcs_match():
    for line in sys.stdin:
        ins = json.loads(line)
        for a in ins['messages']:
            if a['role'] == 'user':
                context = a['content']
            if a['role'] == 'assistant':
                resp = a['content']
        index = context.find('\n根据上面内容回答问题：\n')
        if index == -1:
            continue
        query = context[index+len('\n根据上面内容回答问题：\n'):]
        context = context[:index]
        context = context.replace('其中第1部分文章内容是：', '\x01\x03').replace('其中第2部分文章内容是：', '\x01\x03')\
            .replace('其中第3部分文章内容是：', '\x01\x03').replace('其中第1部分内容是：', '\x01\x03').replace('其中第2部分内容是：', '\x01\x03').replace('其中第3部分内容是：', '\x01\x03')
        parts = context.split('\x01\x03')
        parts = [a for a in parts if len(a)]
        sents = __split_sentences(resp)
        sents = [a for a in sents if len(a) >= 5]
        for sent in sents:
            for part_text in parts:
                score, left, right = similarity(sent, part_text)
                if len(right) > len(sent)/2 and len(right) < len(sent) *2 and score>0.7:
                    start = part_text.find(right)
                    end = start+len(right)
                    end = end+10
                    start =max(0, start-10)
                    spanstr = part_text[start:end]
                    left_punctuation, right_punctuation = find_nearest_punctuation(spanstr,right)
                    if left_punctuation:
                        text = spanstr[left_punctuation+1: right_punctuation]
                        new_index = part_text.find(text)
                        # print(sent, right.replace('\n', '<n>'), spanstr[left_punctuation+1: right_punctuation], score, sep='\t')
                        new = {
                            'title': part_text[:10],
                            "context": part_text,
                            "question": sent,
                            "id": get_md5(text + sent),
                            "answers": {
                                "answer_start": [new_index],
                                "text": [text],
                            },
                        }
                        print(json.dumps(new, ensure_ascii=False))


def lcs_baoqian():
    for line in sys.stdin:
        ins = json.loads(line)
        for a in ins['messages']:
            if a['role'] == 'user':
                context = a['content']
            if a['role'] == 'assistant':
                resp = a['content']
        index = context.find('\n根据上面内容回答问题：\n')
        if index == -1:
            continue
        query = context[index+len('\n根据上面内容回答问题：\n'):]
        context = context[:index]
        context = context.replace('其中第1部分文章内容是：', '\x01\x03').replace('其中第2部分文章内容是：', '\x01\x03')\
            .replace('其中第3部分文章内容是：', '\x01\x03').replace('其中第1部分内容是：', '\x01\x03').replace('其中第2部分内容是：', '\x01\x03').replace('其中第3部分内容是：', '\x01\x03')
        parts = context.split('\x01\x03')
        parts = [a for a in parts if len(a)]
        part_text = random.choice(parts)
        sents = __split_sentences(resp)
        sents = [a for a in sents if len(a) >= 5]
        if len(sents) == 0:
            continue
        sent = sents[0]
        if '抱歉' not in sent:
            continue
        score, left, right = similarity(sent, context)
        if score < 0.7 and score > 0.3:
            new = {
                'title': part_text[:10],
                "context": part_text,
                "question": sent,
                "id": get_md5(part_text + sent),
                "answers": {
                    "answer_start": [],
                    "text": [],
                },
            }
            print(json.dumps(new, ensure_ascii=False))

def lcs_error():
    for line in sys.stdin:
        ins = json.loads(line)
        for a in ins['messages']:
            if a['role'] == 'user':
                context = a['content']
            if a['role'] == 'assistant':
                resp = a['content']
        index = context.find('\n根据上面内容回答问题：\n')
        if index == -1:
            continue
        query = context[index+len('\n根据上面内容回答问题：\n'):]
        context = context[:index]
        context = context.replace('其中第1部分文章内容是：', '\x01\x03').replace('其中第2部分文章内容是：', '\x01\x03')\
            .replace('其中第3部分文章内容是：', '\x01\x03').replace('其中第1部分内容是：', '\x01\x03').replace('其中第2部分内容是：', '\x01\x03').replace('其中第3部分内容是：', '\x01\x03')
        parts = context.split('\x01\x03')
        parts = [a for a in parts if len(a)]
        part_text = random.choice(parts)
        if '\n' not in resp:
            continue
        sent = resp.split('\n')[0]
        if len(sent) < 5 or '。' in sent:
            continue
        score, left, right = similarity(sent, context)
        if score < 0.5 and score > 0.1:
            new = {
                'title': part_text[:10],
                "context": part_text,
                "question": sent,
                "id": get_md5(part_text + sent),
                "answers": {
                    "answer_start": [],
                    "text": [],
                },
            }
            print(json.dumps(new, ensure_ascii=False))

def yindaoyu():
    for line in sys.stdin:
        ins = json.loads(line)
        for a in ins['messages']:
            if a['role'] == 'user':
                context = a['content']
            if a['role'] == 'assistant':
                resp = a['content']
        index = context.find('\n根据上面内容回答问题：\n')
        if index == -1:
            continue
        query = context[index+len('\n根据上面内容回答问题：\n'):]
        # context = context[:index]
        # context = context.replace('其中第1部分文章内容是：', '\x01\x03').replace('其中第2部分文章内容是：', '\x01\x03')\
        #     .replace('其中第3部分文章内容是：', '\x01\x03').replace('其中第1部分内容是：', '\x01\x03').replace('其中第2部分内容是：', '\x01\x03').replace('其中第3部分内容是：', '\x01\x03')
        # parts = context.split('\x01\x03')
        if '，' in resp[:10] and '抱歉' not in resp[:10] and '文章' not in resp[:10]:
            print(query, resp, sep='###')


if __name__ == "__main__":
    import random
    yindaoyu()
    # query = "有效问卷中的各比例构成如下：<n><n>1. 性别结构：男大学生占样本总量的53.8%，女大学生占46.2%"
    # content = "当前，大学生对社会主义核心价值观的认知除了受到个人基本特征、专业学科等微观因素影响外，还与所在大学的层次、类型有关，也跟所在大学的城市和地理区位有关。因此，为了更好地了解大学生对社会主义核心价值观的认知情况，提高调查样本的代表性和广泛性，笔者近年以广东省 25所高校（6 所本科院校、19 所高职专科院校）的在校大学生作为研究对象，共收集网络问卷 2 534 份，其中有效问卷 2 492 份，有效回收率为 98.3%。 在有效问卷中，从性别结构来看，男大学生占样本总量的 53.8%，女大学生占 46.2%，男女比例为 1.16∶1，性别比大致符合社会现实情况；从学历层次来看，本科生占比最高，达到 59.0%，其次 *本文系 2015 年广东省德育创新项目“自媒体视域下大学生社会主义核心价值观教育机制——基于广州地区高校的实证调查”（项目批准号：2015DYYB102）和 2015 年度华南理工大学课题“高校组织与党建工作研究”（项目批准号：j2dbN8150690）的阶段性成果。 DOI:10.16580/j.sxlljydk.2016.08.014 社会主义核心价值观研究  思想理论教育导刊 58 2016 ?1994-2018 China Academic Journal Electronic Publishing House. All rights reserved. http://www.cnki.net年第 8 期 / 总第 212 期 为大专（37.0%），硕士和博士生占 4.0%；在政治面貌上，共青团员为 65.9%，中共党员为 21.0%，群众为 11.8%，其他为 1.3%。总体来看，本次调查涵盖了不同层次的高校，反映了不同性别、学历层次、政治面貌的大学生情况，具有一定的广泛性和代表性。 2. 调研数据分析 （1）认知水平：大学生总体上对社会主义核心价值观内容的认知情况较好，但群体内部有差异。 <n>在被调查的大学生中，完全知晓社会主义核心价值观的内容包括“三个倡导”占 36.2%，部分知晓社会主义核心价值观的占 61.8%，仅有 2%的大学生表示不知道。具体到对社会主义核心价值观每一层面包含具体内容的认知，大学生知晓比例达到六成以上。按认知程度排名从高到低依次是国家层面的价值目标（ 71.2%）、社会层面的价值取向（70.4%）、个人层面的价值准则（62.5%）。 调查结果显示，对社会主义核心价值观内容的认知呈现出群体内部差异化特点。在学历方面，本科生、专科生完全知晓社会主义核心价值观包括“三个倡导”的比例（71.5%）比研究生（含博士生）的比例（40.6%）高 30.9 个百分点。从政治面貌看，党员大学生完全知晓社会主义核心价值观包括“三个倡导”的比例（33%）比群众大学生（25%）高。 （2）认知途径：大学生学习认知核心价值观的途径多样化，课堂传授是最主要途径。 “你是通过什么途径了解或知道社会主义核心价值观?”的调查显示，大学生主要通过课堂传授习得社会主义核心价值观，占比 50.1%；其他依次为：书刊杂志占 49.0%；党员培训占 48.3%；网络媒体占 43.9%；电视广播占 33.5%。 对不同调查对象的认知渠道深入剖析发现，存在性别、学历层次、政治面貌方面的差异。从性别来看，男大学生更倚重书刊杂志，女大学生更倚重课堂传授。男大学生选择课堂传授、书刊杂志、党员培训、网络媒体、电视广播的比例分别为 47.0%、 50.0%、47.8%、42.8%、35.5%，女生则为 52.7%、 47.4%、49.2%、44.1%、35.8%。从学历层次分析， 本科生和博士生倚重网络媒体，硕士生倚重党员培训。本科生选择课堂传授、书刊杂志、党员培训、 网络媒体、电视广播的比例分别是 44.3%、43.9%、 41.2%、46.1%、31.5%，博士生分别为 18.8%、31.3%、 40.6%、46.9%、28.1%，硕士生则为 29.5%、41.0%、 <n>社会主义核心价值观研究<n><n>| 的行动指南。 | 认知是基础，核心价值观教育要在提升大学生的认 |<n>| --- | --- |<n>| 为了解大学生对社会主义核心价值观培育途 | 知深度、促进价值认同上下工夫。 |<n>| 径的看法和态度，本问卷设计了“你认为高校应该 | 首先，课堂教学是人才培养的主要方式。大学 |<n>| 如何开展社会主义核心价值观教育？”据调查结果 | 教师要坚守立德树人的使命，要深入发掘专业课程 |<n>| 显示，排在前三位的是：46.6%的学生认为“增加 | 中的思想政治教育资源，要坚持以情感人，在教学 |<n>| 社会主义核心价值观主题的学校社团活动、社会实践活动”、42.3%的学生认为要“多利用校内论坛、 | 中渗透情感因素，善于运用“同感”，有意识地选择存在于现实生活中、历史文化中的人物和事件， |<n>| 讲座、新闻宣传”、38.1%的学生选择“以讨论、辩论等互动方式改进思想政治理论课、党课教育模式”。可见，大学生希望培育途径更加多样化，对社会实践活动、校内论坛、讲座、新闻宣传和思想 | 通过创设历史情境、生活情境、交往情境等，增强大学生对历史、生活、文化的感受与体验，通过生活经验、情感的传导，形成价值认同。 |<n>| 政治理论课、党课等有较高期待。 | 其次，充分发挥高校思想政治理论课的主渠道作用，将社会主义核心价值观的认同教育融入思想 |<n><n>"
    # example_batch = {
    #     "id": ['0'],
    #     "question": [query],
    #     "context": [content],
    #     "answers": [{"text": [], "answer_start": []}],
    # }
    # all_nbest_json = process(example_batch)
    # print(all_nbest_json)
