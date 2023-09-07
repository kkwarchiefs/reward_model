# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""


import json
import random
import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

_URL = "https://github.com/pluto-junzeng/ChineseSquad/raw/master/squad-zen/"
_URLS = {
    "train": _URL + "train-zen-v1.0.json",
    "dev": _URL + "dev-zen-v1.0.json",
}


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    with open(filepath, encoding="utf-8") as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "")
            suffix = ''
            if 'type' in article:
                suffix = article['type']
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                for qa in paragraph["qas"]:
                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]
                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    obj = {
                        "title": title,
                        "context": context,
                        "question": qa["question"],
                        "id": qa["id"] + suffix,
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answers,
                        },
                    }
                    print(json.dumps(obj, ensure_ascii=False))
