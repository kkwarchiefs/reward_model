from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/summarization_reward_model_pre/checkpoint-58"

RM_tokenizer = AutoTokenizer.from_pretrained(RM_model_path)
RM_model = AutoModelForSequenceClassification.from_pretrained(RM_model_path, num_labels=1)

texts = [["有什么办法去除头发上的虫子", "哈哈", "你好,首先感谢你对我们产品的信任和支持,俗话说:一分价钱一分货。如果您使用西药最少需要2000元,煎服的药最少需要3000元,做手术最少是5000元,而这些毫无疑问,不会对根治你的顽疾有任何帮助。黄天不负有心人,经过老中医多年临床经验总结研制而成的纯中药制剂【发脱宝】对你会有意想不到的收获!祝你早日摆脱烦恼!]?]?]?]? <|endoftext|>,专家提醒大家如有任何疑问敬请咨询我们的在线专家或者打电话进行订购。以下为系统自动调用 发脱宝正确使用方法介绍: 1、用梳子把头发梳通。每天早晚各梳头10次。可边梳边按摩头皮,以增强头部的血液循环。梳头时不要用力过猛以防划破皮肤。梳子最好选用木梳或牛角梳,塑料梳子容易产生静电,会给头发和头皮带来不良刺激。 2、防止脱发。戒烟忌酒。吸烟会使头皮毛细管收缩,从而影响头发的发育生长。热白酒会使头皮产生湿气,引起脱发。烫发吹风要慎重吹风机吹出的热温度达100度,会"]]


for text in texts:
    print(len(text))
    human_inputs = RM_tokenizer(text[0].replace("\n", ""), text[1].replace("<|startofpiece|>", "").replace("<|endofpiece|>", "").replace("<|endoftext|>", "").replace("\n","").replace(" ", ""), truncation=True, max_length=1024, padding="max_length", return_tensors='pt')
    chatgpt_inputs = RM_tokenizer(text[0].replace("\n", ""), text[2].replace("<|startofpiece|>", "").replace("<|endofpiece|>", "").replace("<|endoftext|>", "").replace("\n","").replace(" ", ""), truncation=True, max_length=1024, padding="max_length", return_tensors='pt')
    #print(human_inputs) 
    human_reward = RM_model(input_ids=human_inputs["input_ids"], attention_mask=human_inputs["attention_mask"], token_type_ids=human_inputs["token_type_ids"])
    chatgpt_reward = RM_model(input_ids=chatgpt_inputs["input_ids"], attention_mask=chatgpt_inputs["attention_mask"], token_type_ids=chatgpt_inputs["token_type_ids"])
    print(human_reward)
    print(chatgpt_reward)
    print("*"*10)
