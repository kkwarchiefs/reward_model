

from transformers import pipeline, AutoTokenizer
from transformers import  AutoModelForSeq2SeqLM, AutoModelForSequenceClassification


no_update_device = "cuda:0"
RM_path = "/search/ai/pretrain_models/roberta-base-finetuned-jd-binary-chinese"
senti_tokenizer = AutoTokenizer.from_pretrained(RM_path)
senti_model = AutoModelForSequenceClassification.from_pretrained(RM_path)
sentiment_pipe = pipeline('sentiment-analysis', model=senti_model, tokenizer=senti_tokenizer, device=no_update_device)


prompt_demo = ["房间非常大，", "小本子不错，", "性价比高，", "比较老的酒店，", "屏幕小了一点，", "地理位置不错", "配件太少了", "收到本书，非常兴奋", "外观漂亮，", "机器不错。"]
sft_response = ["装修很精致,房间是暖色调的房间,白色系的家具搭配", "就喜欢, 胜过一万万努力的闹钟  我选择的的艺术之路", "同价位下马斯克再度亮眼 昨天(3月13日),", "找个安静的环境住着 feel生活指南 1、上海金泽路桥", "是否会更好? 魅族对全面屏一直有着自己的执念,尽管在前", ",体检科,四十年代医院,城市里的别墅;亢河两岸", "苦学还是半桶水 来源:广州日报ied是否越", "把它写下来。同时会得到无比的快乐", "价格实惠。车身同级别中动力的销量之王马赛人破", "左边有显示风量,亮度非常好!~~~~ 宝贝很好,声音很小"]
ppo_response = ["高1.8米居然住了两个人,真心难怪评价这么差!", "但是不行, 他让我放正, 结果后来就开骂了", "真心不如错选 坦白地说,从未听商家或者商家卖家", "服务很差,没有必要的物品随意丢,厨师服务很差,", "质量一般,摄像头更不合格(非专业,现场取景无", ",但是环境更差了,和 ⁇ 灞一样!作为 ⁇ 灞桥头", ",也没法生存 透气护耳帽和厚带不紧,", "顿时长吟一声呜呼,答不得", "内饰乱糟糟,质量不咋样,三大缺点被网友嘲讽", "但是手机电路质量真心不敢恭维,纯铝合金材质几乎是无法"]
rm_texts_temp = [r+p for r,p in zip(prompt_demo, ppo_response)]
pipe_outputs = sentiment_pipe(rm_texts_temp)
for text_score in pipe_outputs:
    score = text_score["score"]
    if "positive" in text_score["label"]:
        score = 1-score
    print(score)
