import argparse
import os
from string import Template

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()


def cut_doc_plus(tokenizer, data, piece_len=750):
    tokens = tokenizer(data)
    tokens_ids = tokens['input_ids'][1:-1]
    index = 0
    piece_data = []
    last_index = 0
    while index < len(tokens_ids):
        index += piece_len
        if index < len(tokens_ids):
            temp_data = tokenizer.decode(tokens_ids[last_index: index])
        else:
            temp_data = tokenizer.decode(tokens_ids[last_index:])
        piece_data.append(temp_data)
        last_index = index
    return piece_data

model_name = "RM_10b_onnx"
device = torch.device('cuda:7')
aa = "。元祐六年（1091年），他⼜被召回朝。不久即元祐六年⼋⽉，⼜因为政⻅不合，调往颍州任知州。元祐七年（1092年）⼆⽉，任扬州知州。元祐⼋年（1093年）九⽉，任定州知州。是年⾼太后去世，哲宗执政，新党再度执政。绍圣元年（1094年）六⽉，贬为宁远军节度副使，再次被贬⾄惠州（今⼴东惠州）。绍圣四年（1097年），年已六⼗⼆岁的苏轼被⼀叶孤⾈送到了徼边荒凉之地海南岛儋州。据说在宋朝，放逐海南是仅⽐满⻔抄斩罪轻⼀等的处罚。他把儋州当成了⾃⼰的第⼆故乡，“我本儋⽿⽒，寄⽣⻄蜀州”。他在这⾥办学堂，介学⻛，以致许多⼈不远千⾥，追⾄儋州，从苏轼学。在宋代⼀百多年⾥，海南从没有⼈进⼠及第。但苏轼北归不久，这⾥的姜唐佐就举乡贡。为此苏轼题诗：“沧海何曾断地脉，珠崖从此破天荒。”⼈们⼀直把苏轼看作是儋州⽂化的开拓者、播种⼈，对他怀有深深的崇敬。在儋州流传下来的东坡村、东坡井、东坡⽥、东坡路、东坡桥、东坡帽等等，表达了⼈们的缅怀之情，连语⾔都有⼀种“东坡话”。身逝常州宋徽宗即位后，苏轼相继被调为廉州安置、舒州团练副使、永州安置。元符三年四⽉（11括各种⽂学样式，他本⼈的创作⼜没有固定不变的规范可循，所以苏⻔的作家在创作上各具⾯⽬。⻩庭坚、陈师道⻓于诗，秦观⻓于词，李廌以古⽂名世，张、晁则诗⽂并擅。同时，他们的艺术⻛貌也各具个性，例如⻩诗⽣新，陈诗朴拙，⻛格都不类苏诗，后来⻩、陈还另外开宗⽴派。苏轼的作品在当时就驰名遐迩，在辽国、⻄夏等地都⼴受欢迎。北宋末年，朝廷⼀度禁⽌苏轼作品的流传，但是禁愈严⽽传愈⼴。到了南宋党禁解弛，苏轼的集⼦⼜以多种版本⼴为流传，以后历代翻刻不绝。在后代⽂⼈的⼼⽬中，苏轼是⼀位天才的⽂学巨匠，⼈们争相从苏轼的作品中汲取营养。在⾦国和南宋对峙的时代，苏轼在南北两⽅都发⽣了深远的影响。苏诗不但影响有宋⼀代的诗歌，⽽且对明代的公安派诗⼈和清初的宋诗派诗⼈有要的启迪。苏轼的词体解放精神直接为南宋⾟派词⼈所继承，形成了与婉约词平分秋⾊的豪放词派，其影响⼀直波及清代陈维崧等⼈。苏轼的散⽂，尤其是他的⼩品⽂，是明代标举抒性灵的公安派散⽂的艺术渊源，直到清代袁枚、郑燮的散⽂中仍可时⻅苏⽂的影响。苏轼还以和蔼可亲、幽默机智的形象留存在后代普通⼈⺠⼼⽬中。他在各地的游踪，他在⽣活中的各种发明都是后⼈喜爱的话题。在宋代作家中，就受到后⼈⼴泛喜爱的程度⽽⾔，苏轼是⽆与伦⽐的。历代评价赵祯：吾今⼜为吾⼦孙得太平宰相两⼈。（陈鹄《耆旧续闻·卷⼆》引，俞⽂豹《吹剑录》作“吾为⼦孙得两相”。）王辟之：⼦瞻⽂章议论，出当世，⻛格⾼迈。据《坛经》记载，惠能初⻅五祖弘忍时，弘忍⼤师说：“汝是岭南⼈，⼜是獦獠，若为堪作佛？”惠能答道：“⼈虽有南北，佛性本⽆南北，獦獠身与和尚不同，佛性有何差别？”苏轼接受了这种“南北不⼆”的观念，并在《闻潮阳吴⼦出家》⼀诗中表达了“当为⼦吼，佛法⽆南北”的思想。在此诗之前，他在《送⼩本禅师赴法云》中也曾有“是身如浮云，安得限南北”的说法。在“南北不⼆”观念的影响下，苏轼逐渐将南北融为⼀体，这在他的诗歌中多有表露，如“⼈间底处有南北，纷纷鸿雁何曾冥”、“⽚云会得⽆⼼否，南北东⻄只⼀天”。既然“南北”本⽆分别，那么随⼼适意的⽣活状态便成了苏轼的⼈⽣追求，所谓“我⾏⽆南北，适意乃所祈”。"
bb = "directly, because those two methods will change thewhole architecture of pretrained model, resultingslightly worse results compared with original pretrained model. As result, the integration of CLH3Gand pretrained Sequence to Sequence models requires abundant H3G data to achieve comparableresults.AcknowledgementsWe would like to thank the anonymous reviewersfor their constructive comments.ReferencesXiang Ao, Xiting Wang, Ling Luo, Ying Qiao, QingHe, and Xing Xie. 2021. Pens: A dataset and genericframework for personalized news headline generation. In Proceedings of the 59th Annual Meeting ofthe Association for Computational Linguistics andthe 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers),pages 82–92.Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointlylearning to align and translate. arXiv preprintarXiv:1409.0473.Hengyi Cai, Hongshen Chen, Yonghao Song, ZhuoyeDing, Yongjun Bao, Weipeng Yan, and Xiaofang Zhao. 2020. Group-wise contrastive learning for neural dialogue generation. arXiv preprintarXiv:2009.07543.Ting Chen, Simon Kornblith, Mohammad Norouzi, andGeoffrey Hinton. 2020. A simple framework forcontrastive learning of visual representations. In International conference on machine learning, pages1597–1607. PMLR.Ning Dai, Jianze Liang, Xipeng Qiu, and XuanjingHuang. 2019. Style transformer: Unpaired text styletransfer without disentangled latent representation.arXiv preprint arXiv:1905.05621.Jacob Devlin, Ming-Wei Chang, Kenton Lee, andKristina Toutanova. 2018. Bert: Pre-training of deepbidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.Bonnie Dorr, David Zajic, and Richard Schwartz. 2003.Hedge trimmer: A parse-and-trim approach to headline generation. Technical report, MARYLANDUNIV COLLEGE PARK INST FOR ADVANCEDCOMPUTER STUDIES.Daniil Gavrilov, Pavel Kalaidin, and Valentin Malykh.2019. Self-attentive model for headline generation.In European Conference on Information Retrieval,pages 87–93. Springer.Di Jin, Zhijing Jin, Joey Tianyi Zhou, Lisa Orii, and Peter Szolovits. 2020. Hooks in the headline: Learningto generate headlines with controlled styles. arXivpreprint arXiv:2004.01980.Guillaume Lample, Sandeep Subramanian, Eric Smith,Ludovic Denoyer, Marc’Aurelio Ranzato, and Y-LanBoureau. 2018. Multiple-attribute text rewriting. InInternational Conference on Learning Representations.Seanie Lee, Dong Bok Lee, and Sung Ju Hwang.2020. Contrastive learning with adversarial perturbations for conditional text generation. arXiv preprintarXiv:2012.07280.Chin-Yew Lin. 2004. Rouge: A package for automaticevaluation of summaries. In Text summarizationbranches out, pages 74–81.Dayiheng Liu, Yeyun Gong, Jie Fu, Wei Liu, Yu Yan,Bo Shao, Daxin Jiang, Jiancheng Lv, and Nan Duan.2020. Diverse, controllable, and keyphrase-aware: Acorpus and method for news multi-headline generation. arXiv preprint arXiv:2004.03875.Yixin Liu and Pengfei Liu. 2021. Simcls: A simpleframework for contrastive learning of abstractivesummarization. arXiv preprint arXiv:2106.01890.Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the40th annual meeting of the Association for Computational Linguistics, pages 311–318.Colin Raffel, Noam Shazeer, Adam Roberts, KatherineLee, Sharan Narang, Michael Matena, Yanqi Zhou,Wei Li, and Peter J Liu. 2019. Exploring the limitsof transfer learning with a unified text-to-text transformer."
RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/summarization_reward_model/checkpoint-58/"
RM_model_path = '/search/ai/pretrain_models/roberta-base-finetuned-jd-binary-chinese'
RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/reward_model_glm_10b_bak/final"
RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/reward_model_glm_10b/checkpoint-955"
RM_model_path = "./output/rm_model/"
RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/reward_model_glm_10b/checkpoint-3439"

RM_tokenizer = AutoTokenizer.from_pretrained(RM_model_path, use_fast=True, trust_remote_code=True)
sentence = RM_tokenizer(aa)
print(len(sentence['input_ids']))
sentence = RM_tokenizer(bb)
print(len(sentence['input_ids']))
# print(cut_doc_plus(RM_tokenizer, aa, 100))
# print(cut_doc_plus(RM_tokenizer, bb, 100))
# print(RM_tokenizer.decode(sentence['input_ids']))
model_path = "/search/ai/pretrain_models/infoxlm-base/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sentence = tokenizer(aa)
print(len(sentence['input_ids']))
sentence = tokenizer(bb)
print(len(sentence['input_ids']))
# config = AutoConfig.from_pretrained(
# 		RM_model_path,
#         trust_remote_code=True
#     )
# config.classifier_dropout = 0.1
# config.num_labels = 1
# RM_model = AutoModelForSequenceClassification.from_pretrained(RM_model_path, config=config, trust_remote_code=True)
# # model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model.bin'), map_location="cpu")
# # new_model_dict = {k.replace('hf_model.', ''): v for k, v in model_dict.items()}
# # load_result = RM_model.load_state_dict(new_model_dict, strict=True)
#
# RM_model = RM_model.half().to(device)
# query_text = '什么人不能喝三七粉' + "[UNUSED1]"
# response_text = '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。'
# input = RM_tokenizer(query_text + response_text, max_length=512, truncation=True, return_tensors="pt").to(device)
# print(input)
# # print(RM_model(input['input_ids'], input['attention_mask']))
# RM_model = RM_model.eval()  # 转换为eval模式
# inputs = (input['input_ids'], input['attention_mask'])  # 模型测试输入数据
# os.makedirs(f"model_store/{model_name}/1", exist_ok=True)
# torch.onnx.export(
# 	RM_model,
# 	inputs,
# 	f"model_store/{model_name}/1/model.onnx",  # 输出模型文件名
# 	input_names=['input_ids', 'attention_mask'],  # 输入节点名，每一个名称对应一个输入名
#     output_names=['output'],  # 输出节点名，每一个名称对应一个输出名
# 	opset_version=14,
# 	dynamic_axes={'input_ids': {0: 'B', 1: 'C'}, 'attention_mask': {0: 'B', 1: 'C'}}  # 声明动态维度，默认输入维度固定，可以声明为可变维度（用字符串占位符表示）
# )


# traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")



