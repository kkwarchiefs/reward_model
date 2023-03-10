from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
import datetime

path = "/search/ai/kaitongyang/online/model/GLM-10B-chinese-customization_02-28-11-26/30630"
path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/cur_model/30630"
path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/RLHF_MODEL/120"
#path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/RLHF_MODEL_PRE/60"
device = "cuda:3"
suffix = " [回答][gMASK]"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
model = model.half().to(device)
model.eval()

print("load model and tokenizer done !!!")


result = []
data_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/data/rm_data_0301-gpt3.5-8w/success-x02.json"
datas = open(data_path).read().splitlines()
for data in datas[:20]:
    example = json.loads(data)
    prompt = example["prompt"]
    chatgpt = response = example["response"]
    inputs = tokenizer(prompt + suffix, return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key][:,:-1]
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    inputs = inputs.to(device)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id, num_beams=4, no_repeat_ngram_size=7, repetition_penalty=1.1, min_length=3)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    response = tokenizer.decode(outputs.squeeze()[inputs["input_ids"].size()[1]:].tolist())
    print(response)
    result.append({"prompt":prompt, "gpt_0.4":response, "chatgpt":chatgpt})

print(result)


with open(os.path.join("/search/ai/kaitongyang/RLHF_DEBUG/RM/data", "train.jsonl"), "w", encoding='utf-8') as out:
    for demo in result:
        json.dump(demo, out, ensure_ascii=False)
        out.write("\n")

    
