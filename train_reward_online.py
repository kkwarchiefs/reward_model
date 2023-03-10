from datasets import load_dataset
import random
import os
from datasets import Dataset
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
    HfArgumentParser,
)
import torch
from transformers.utils import PaddingStrategy
from typing import Optional, Union, List, Dict, Any
import evaluate
from dataclasses import dataclass, field
import torch.nn as nn
import numpy as np
import pandas as pd
import json


def process_tokenizer(datas):

    input_ids_one_list = []
    attention_mask_one_list = []
    token_type_ids_one_list = []
   
    input_ids_two_list = []
    attention_mask_two_list = []
    token_type_ids_two_list = []
 
    input_ids_three_list = []
    attention_mask_three_list = []
    token_type_ids_three_list = []
 

    input_ids_four_list = []
    attention_mask_four_list = []
    token_type_ids_four_list = []
 

    for data in datas:
        prompt, response_list = data

        tokenized_one = tokenizer(prompt, response_list[0].replace("\n", ""), truncation=True, max_length=512, padding="max_length")
        input_ids_one_list.append(tokenized_one["input_ids"])
        attention_mask_one_list.append(tokenized_one["attention_mask"])
        token_type_ids_one_list.append(tokenized_one["token_type_ids"])


        tokenized_two = tokenizer(prompt, response_list[1].replace("\n", ""), truncation=True, max_length=512, padding="max_length")
        input_ids_two_list.append(tokenized_two["input_ids"])
        attention_mask_two_list.append(tokenized_two["attention_mask"])
        token_type_ids_two_list.append(tokenized_two["token_type_ids"])


        tokenized_three = tokenizer(prompt, response_list[2].replace("\n", ""), truncation=True, max_length=512, padding="max_length")
        input_ids_three_list.append(tokenized_three["input_ids"])
        attention_mask_three_list.append(tokenized_three["attention_mask"])
        token_type_ids_three_list.append(tokenized_three["token_type_ids"])

        tokenized_four = tokenizer(prompt, response_list[3].replace("\n", ""), truncation=True, max_length=512, padding="max_length")
        input_ids_four_list.append(tokenized_four["input_ids"])
        attention_mask_four_list.append(tokenized_four["attention_mask"])
        token_type_ids_four_list.append(tokenized_four["token_type_ids"])
        
    result = Dataset.from_dict({'input_ids_one':input_ids_one_list, 'attention_mask_one':attention_mask_one_list, 'token_type_ids_one':token_type_ids_one_list, 'input_ids_two':input_ids_two_list, 'attention_mask_two':attention_mask_two_list, 'token_type_ids_two':token_type_ids_two_list, 'input_ids_three':input_ids_three_list, 'attention_mask_three':attention_mask_three_list, 'token_type_ids_three':token_type_ids_three_list, 'input_ids_four':input_ids_four_list, 'attention_mask_four':attention_mask_four_list, 'token_type_ids_four':token_type_ids_four_list})
    return result
 

def load_dataset(data_path, train_size=2500, top_k=4, sample=4): 
    ins = pd.read_csv(data_path)
    result = []
    count = 0
    for v in ins.values:
        prompt = v[1]
        response_list = json.loads(v[3])
        response_dict = dict()
        for candidate in response_list:
            response_dict[int(candidate['level'])] = candidate['name']
        response_dict = sorted(response_dict.items(), key = lambda kv:kv[0], reverse=True)
        temp_response_dict = dict()
        for i in response_dict:
            level, response = i
            if level in temp_response_dict:
                continue
            else:
                temp_response_dict[level] = response
        if len(temp_response_dict) < top_k:
            continue

        index_result = []
        for _ in range(sample):
            index = sorted(random.sample(list(range(len(temp_response_dict))), top_k))
            if index in index_result:
                continue
            index_result.append(index)
        for index_list in index_result:
            temp_index = 0
            temp_result = []
            for i in temp_response_dict:
                if temp_index in index_list:
                    temp_result.append(temp_response_dict[i])
                temp_index = temp_index + 1
            result.append([prompt,temp_result])       

    train_dataset = process_tokenizer(result[:train_size])
    dev_dataset = process_tokenizer(result[train_size:]) 
    
    return train_dataset, dev_dataset



# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=0, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": "If you want to resume training where it left off."}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[int] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="/search/ai/pretrain_models/models--nghuyong--ernie-3.0-base-zh",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default="5",
        metadata={
            "help": "The number of training epochs for the reward model. OpenAI used 5."
        }
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir=f"reward_model",
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
)

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(script_args.model_name, num_labels=1)
# Need to do this for gpt2, because it doesn't have an official pad token.
#tokenizer.pad_token = tokenizer.eos_token
#model.config.pad_token_id = tokenizer.eos_token_id

num_proc = (
    8
)  # Can adjust to be higher if you have more processors. Should work even if you don't have 8 CPUs, though.


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_one = []
        features_two = []
        features_three = []
        features_four = []
        for feature in features:
            features_one.append({"input_ids": feature["input_ids_one"], "attention_mask": feature["attention_mask_one"], "token_type_ids":feature["token_type_ids_one"]})
            features_two.append({"input_ids": feature["input_ids_two"], "attention_mask": feature["attention_mask_two"], "token_type_ids":feature["token_type_ids_two"]})
            features_three.append({"input_ids": feature["input_ids_three"], "attention_mask": feature["attention_mask_three"], "token_type_ids":feature["token_type_ids_three"]})
            features_four.append({"input_ids": feature["input_ids_four"], "attention_mask": feature["attention_mask_four"], "token_type_ids":feature["token_type_ids_four"]})
        batch_one = self.tokenizer.pad(
            features_one,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        batch_two = self.tokenizer.pad(
            features_two,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        batch_three = self.tokenizer.pad(
            features_three,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
    
        batch_four = self.tokenizer.pad(
            features_four,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
    
        batch = {
            "input_ids_one": batch_one["input_ids"],
            "attention_mask_one": batch_one["attention_mask"],
            "token_type_ids_one":batch_one["token_type_ids"],
            "input_ids_two": batch_two["input_ids"],
            "attention_mask_two": batch_two["attention_mask"],
            "token_type_ids_two":batch_two["token_type_ids"],
            "input_ids_three": batch_three["input_ids"],
            "attention_mask_three": batch_three["attention_mask"],
            "token_type_ids_three":batch_three["token_type_ids"],
            "input_ids_four": batch_four["input_ids"],
            "attention_mask_four": batch_four["attention_mask"],
            "token_type_ids_four":batch_four["token_type_ids"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        #https://github.com/huggingface/transformers/issues/7848
        input_ids = torch.cat([inputs["input_ids_one"], inputs["input_ids_two"],inputs["input_ids_three"], inputs["input_ids_four"]], dim=0)
        attention_mask = torch.cat([inputs["attention_mask_one"], inputs["attention_mask_two"], inputs["attention_mask_three"], inputs["attention_mask_four"]], dim=0)
        token_type_ids = torch.cat([inputs["token_type_ids_one"], inputs["token_type_ids_two"], inputs["token_type_ids_three"], inputs["token_type_ids_four"]], dim=0)
        #rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        #rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        emb = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        rewards_one, rewards_two, rewards_three, rewards_four  = emb.logits.chunk(4, dim=0)
        loss_1 = -nn.functional.logsigmoid(rewards_one - rewards_two).mean()
        loss_2 = -nn.functional.logsigmoid(rewards_two - rewards_three).mean()
        loss_3 = -nn.functional.logsigmoid(rewards_three - rewards_four).mean()
        loss = (loss_1 + loss_2 + loss_3)/3 
        if return_outputs:
            return loss, {"rewards_one": rewards_one, "rewards_two": rewards_two, "rewards_three": rewards_three, "rewards_four": rewards_four}
        return loss

dataset_path="/search/ai/kaitongyang/RLHF_DEBUG/RM/data/reward_data/reward_model_data_sql_result_20230310111430.csv"
train_dataset, eval_dataset = load_dataset(dataset_path)

print("train_dataset num : " + str(len(train_dataset)))
print("eval_dataset num : " + str(len(eval_dataset)))


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train(script_args.resume_from_checkpoint)

# Push to the hub so you can share it with people :D
reward_path = "reward_model"
model.save_pretrained(os.path.join(reward_path, script_args.model_name))
tokenizer.save_pretrained(os.path.join(reward_path, script_args.model_name))
