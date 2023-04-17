import argparse
import os
from string import Template

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering
from modeling_cpt import CPTForSequenceClassification, CPTForQuestionAnswering
from transformers import BertTokenizer
parser = argparse.ArgumentParser()

model_name = "QuestionAnswering_onnx"
device = torch.device('cuda:0')



RM_model_path = "/search/ai/jamsluo/GLM_RLHF/reward_model/output/squad_cmrc_du_wiki_nyt/"

tokenizer = AutoTokenizer.from_pretrained(RM_model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(
		RM_model_path,
	use_fast=True,
	trust_remote_code=True
    )
model = AutoModelForQuestionAnswering.from_pretrained(RM_model_path, config=config, trust_remote_code=True)
model = model.to(device)
question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"
pad_on_right = tokenizer.padding_side == "right"
max_seq_length =384


def prepare_train_features(examples):
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
		padding= True,
	)

	# Since one example might give us several features if it has a long context, we need a map from a feature to
	# its corresponding example. This key gives us just that.
	sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
	# The offset mappings will give us a map from token to character position in the original context. This will
	# help us compute the start_positions and end_positions.
	offset_mapping = tokenized_examples.pop("offset_mapping")

	# Let's label those examples!
	tokenized_examples["start_positions"] = []
	tokenized_examples["end_positions"] = []

	for i, offsets in enumerate(offset_mapping):
		# We will label impossible answers with the index of the CLS token.
		input_ids = tokenized_examples["input_ids"][i]
		cls_index = input_ids.index(tokenizer.cls_token_id)

		# Grab the sequence corresponding to that example (to know what is the context and what is the question).
		sequence_ids = tokenized_examples.sequence_ids(i)

		# One example can give several spans, this is the index of the example containing this span of text.
		sample_index = sample_mapping[i]
		answers = examples[answer_column_name][sample_index]
		# If no answers are given, set the cls_index as answer.
		if len(answers["answer_start"]) == 0:
			tokenized_examples["start_positions"].append(cls_index)
			tokenized_examples["end_positions"].append(cls_index)
		else:
			# Start/end character index of the answer in the text.
			start_char = answers["answer_start"][0]
			end_char = start_char + len(answers["text"][0])

			# Start token index of the current span in the text.
			token_start_index = 0
			while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
				token_start_index += 1

			# End token index of the current span in the text.
			token_end_index = len(input_ids) - 1
			while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
				token_end_index -= 1

			# Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
			if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
				tokenized_examples["start_positions"].append(cls_index)
				tokenized_examples["end_positions"].append(cls_index)
			else:
				# Otherwise move the token_start_index and token_end_index to the two ends of the answer.
				# Note: we could go after the last offset if the answer is the last word (edge case).
				while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
					token_start_index += 1
				tokenized_examples["start_positions"].append(token_start_index - 1)
				while offsets[token_end_index][1] >= end_char:
					token_end_index -= 1
				tokenized_examples["end_positions"].append(token_end_index + 1)

	return tokenized_examples
examples = {
"question": ["历史一阵中的小前锋是谁？", "历史一阵中的小前锋是谁？"],
"context": ["美媒评历史五大最佳阵容：奥尼尔三阵，库里科比二阵，一阵无悬念v阿成嘞2023年3月29日13:49山东体育领域创作者说起历史最佳阵容很多媒体认为是乔佛魔沙皇，而这五人就是乔丹、詹姆斯、魔术师、邓肯和奥尼尔，但这似乎对于历史地位最高的中锋贾巴尔并不友好，他是历史第三，结果历史最佳阵容的位置却被奥尼尔给抢了去，值得一提的是近期美媒评出了历史上前五套最佳阵容，其中奥尼尔跌到了三阵，大家一起来看一下合理吗VP，关键是他还有2届最佳防守球员奖以及历史盖帽王这样的荣誉。历史一阵图片这一阵基本大家都清楚是谁了，控卫是魔术师约翰逊，魔术师之所以能够成为历史第一控卫是因为他拥有3个FMVP奖杯，历史控卫最多，同时魔术师场均助攻11.2次历史最多，魔术师生涯12年里带队9次打进了总决赛5次拿下了总冠军，他的统治力非常出众，如果不是因为艾滋病，魔术师本可以成为乔丹之前的历史第一人。分卫是乔丹，这里无需多言，历史上拿下得分王最多的球员，而且巅峰十年里10进总决赛、9次最佳一防，历史上攻防一体最好的球员，同时他常规赛场均30.12分历史第一、季后赛场均33.4分历史第一。小前锋是詹姆斯这个也无需多言，如果说乔丹是惊艳、无解，那么詹姆斯就是细水长流让人无法企及，比如连续得分上双纪录、最年轻得分纪录、3万+1万+1万纪录以及历史总得分王。图片大前锋邓肯，邓肯实际上在03年夺冠之后，他就可以问鼎历史第一大前锋了，但考虑到马龙是当时历史总得分第二，再加上邓肯又不迎合媒体，这也导致他一直没有受到认可，直到邓肯生涯退役之后，他才得到了历史第一大前锋的位置，他太低调了以至于即便单核带队做到过夺冠，还有5冠3FMVP和",
			"美媒评历史五大最佳阵容：奥尼尔三阵，库里科比二阵，一阵无悬念v阿成嘞2023年3月29日13:49山东体育领域创作者说起历史最佳阵容很多媒体认为是乔佛魔沙皇，而这五人就是乔丹、詹姆斯、魔术师、邓肯和奥尼尔，但这似乎对于历史地位最高的中锋贾巴尔并不友好，他是历史第三，结果历史最佳阵容的位置却被奥尼尔给抢了去，值得一提的是近期美媒评出了历史上前五套最佳阵容，其中奥尼尔跌到了三阵，大家一起来看一下合理吗火拿下了两个总冠军，历史上盖帽最多的后卫。小前锋杜兰特，杜兰特作为历史第三锋线，他进入这套阵容没有太大的疑问，杜兰特是历史级别的单打手，他拥有着无差别单打的能力，他是历史上拿下得分王最年轻的球员，生涯也是拿下过4届得分王，杜兰特还在总决赛打出了180的三投命中率，即便现在已经35岁，但杜兰特仍旧是现役联盟关键时刻的进攻第一选择。图片大前锋是诺维茨基，诺天王虽然生涯只有1个总冠军，但他进入这份榜单没有太大的疑问，毕竟他曾拿下过常规赛MVP证明了自己，此外他还单核带队拿下了总冠军，生涯总得分历史第六，诺天王还是一人一城效力时间最长的球员，诺天王是外籍球员的第一人，因为他才让更多的欧洲球员敢于登陆NBA来证明自己。中锋是奥尼尔，作为历史上最具统治力的中锋，奥尼尔排在了第三的位置确实是让人没有想到，看到大鲨鱼排在第三估计很多球迷都想看前二到底是谁了，奥尼尔是除了乔丹之外第二个做到三连FMVP的球员，而且总决赛末节场均11.4分也是历史最高纪录，奥尼尔生涯还拿下了1个常规赛MVP和2届得分王。历史最佳二阵图片这一套也被认为是能够和历史第一阵容媲美的存在，控卫是库里，库里创造了小球时代，对于篮球的"],
"answers": [{ "text": [], "answer_start": [] }, { "text": [], "answer_start": [] }],
}
tokenized_examples = prepare_train_features(examples)
print(tokenized_examples)
# print(RM_model(input['input_ids'], input['attention_mask']))
model = model.eval()  # 转换为eval模式
inputs = (tokenized_examples['input_ids'], tokenized_examples['attention_mask'], tokenized_examples['start_positions'], tokenized_examples['end_positions'])  # 模型测试输入数据
tensor_inputs = [torch.tensor(a,dtype=torch.int64).to(device) for a in inputs ]
tensor_inputs = (tensor_inputs[0], tensor_inputs[1])
print(model(tensor_inputs[0], tensor_inputs[1]))
os.makedirs(f"model_store/{model_name}/1", exist_ok=True)
torch.onnx.export(
	model,
	tensor_inputs,
	f"model_store/{model_name}/1/model.onnx",  # 输出模型文件名
	input_names=['input_ids', 'attention_mask'],  # 输入节点名，每一个名称对应一个输入名
    output_names=['start_logits', 'end_logits'],  # 输出节点名，每一个名称对应一个输出名
	opset_version=14,
	dynamic_axes={'input_ids': {0: 'B', 1: 'C'}, 'attention_mask': {0: 'B', 1: 'C'}}  # 声明动态维度，默认输入维度固定，可以声明为可变维度（用字符串占位符表示）
)


# traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")



