from datasets import load_dataset
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

def load_dataset(data_path, datase_type):
    datas = open(data_path).read().splitlines()
    input_ids_j_list = []
    attention_mask_j_list = []
    token_type_ids_j_list = []
    input_ids_k_list = []
    attention_mask_k = []
    token_type_ids_k_list = []
    if datase_type == "train":
        datas = datas[:15000]
    else:
        datas = datas[15000:]
    for data in datas:
        example = json.loads(data)
        tokenized_j = tokenizer(example["prompt"].replace("\n", ""), example["response"].replace("\n", ""), truncation=True, max_length=1024, padding="max_length")
        tokenized_k = tokenizer(example["prompt"].replace("\n", ""), example["human"].replace("\n", ""), truncation=True, max_length=1024, padding="max_length")
        #print(tokenized_j)
        #print(text_j)
        #print(tokenizer.decode(tokenized_j["input_ids"]))
        input_ids_j_list.append(tokenized_j["input_ids"])
        attention_mask_j_list.append(tokenized_j["attention_mask"])
        token_type_ids_j_list.append(tokenized_j["token_type_ids"])
        input_ids_k_list.append( tokenized_k["input_ids"])
        attention_mask_k.append(tokenized_k["attention_mask"])
        token_type_ids_k_list.append(tokenized_k["token_type_ids"])
    result = Dataset.from_dict({'input_ids_j':input_ids_j_list, 'attention_mask_j':attention_mask_j_list, 'token_type_ids_j':token_type_ids_j_list, 'input_ids_k':input_ids_k_list, 'attention_mask_k':attention_mask_k, 'token_type_ids_k':token_type_ids_k_list})
    return result
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
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=8)
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
    output_dir=f"summarization_reward_model",
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
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"], "token_type_ids":feature["token_type_ids_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"], "token_type_ids":feature["token_type_ids_k"]})
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "token_type_ids_j":batch_j["token_type_ids"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "token_type_ids_k":batch_k["token_type_ids"],
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
        input_ids = torch.cat([inputs["input_ids_j"], inputs["input_ids_k"]], dim=0)
        attention_mask = torch.cat([inputs["attention_mask_j"], inputs["attention_mask_k"]], dim=0)
        token_type_ids = torch.cat([inputs["token_type_ids_j"], inputs["token_type_ids_k"]], dim=0)
        #rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        #rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        emb = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        rewards_j, rewards_k  = emb.logits.chunk(2, dim=0)
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

train_dataset_path="data/success-0223.json"
eval_dataset_path="data/success-0223.json"

train_dataset = load_dataset(train_dataset_path, "train")
eval_dataset = load_dataset(eval_dataset_path, "eval")

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
reward_path = "summarization_reward_model"
model.save_pretrained(os.path.join(reward_path, script_args.model_name))
tokenizer.save_pretrained(os.path.join(reward_path, script_args.model_name))
