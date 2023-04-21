import numpy as np
import faiss
import torch
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer
from create_countext import SearchContext, QAContext

class ChatPdf():
    def __init__(self,inference_function, content_type="web page"):
        self.content_type = content_type
        if self.content_type == "web page":
            self.content_name = "网页"
        elif self.content_type == "PDF file":
            self.content_name = "PDF文件"
        self.glm_inference = inference_function
        self.create_context = QAContext()

    def read_data(self, file_path):
        text = open(file_path).read()
        # self.summary = self.get_summary(text)
        # self.questions = self.get_question(text)
        self.create_context.refresh_data(text)

    def get_summary(self, data):
        max_len = 1600
        piece_data = self.cut_doc(data, piece_len=750, single_piece_max_len=max_len)
        final_sum_content = []
        if len(piece_data) == 1:
            final_sum_content.append("第1部分{}内容：".format(self.content_name) + piece_data[0])
        else:
            for i in range(len(piece_data)):
                piece_content = []
                for j in range(max(i - 2, 0), i):
                    piece_content.append("第{}部分{}内容：".format(j + 1, self.content_name) + piece_data[j])
                piece_content.append("第{}部分{}内容：".format(i + 1, self.content_name) + piece_data[i])
                piece_content.append("生成第{}部分{}内容的摘要".format(i + 1, self.content_name))
                summ = self.glm_inference("<n>".join(piece_content).replace("\n", "<n>"))
                final_sum_content.append("第{}部分的主要内容是：".format(i + 1) + summ)
                if len(" ".join(final_sum_content))-20>=max_len:
                    print ("total sum max len")
                    break
        final_sum_content.append("根据上面各个部分内容的介绍生成整个{}的摘要".format(self.content_name))
        summ = self.glm_inference("<n>".join(final_sum_content).replace("\n", "<n>"))
        return summ

    def get_question(self, data):
        piece_data = []
        piece_len = 600
        if len(data) < piece_len * 3:
            input_word = data
        else:
            out_word_len = int((len(data) - piece_len * 3) / 4)
            input_word = data[out_word_len:out_word_len + piece_len] + \
                         data[out_word_len * 2 + piece_len: out_word_len * 2 + piece_len * 2] + \
                         data[out_word_len * 3 + piece_len * 2: out_word_len * 3 + piece_len * 3]
        input_word = input_word + "针对上面内容生成三个问题，结果使用类似的[\"<question>\",\"<question>\",\"<question>\"]json展示，不要生成类似于\"这个{}主要讲了什么内容?\"这种类似的问题".format(self.content_name)
        questions = self.glm_inference(input_word.replace("\n", "<n>"))
        return questions

    def get_mrc(self, query):
        query_context = self.create_context.get_query_context(query)
        #print ("检索上下文：", query_context)
        # sft_prompt = "整个%s的摘要是：%s<n>" % (self.content_name, self.summary)
        # sft_prompt = "整个%s的摘要是：%s<n>" % (self.content_name, "")
        sft_prompt = ''
        for index, part in enumerate(query_context):
            sft_prompt = sft_prompt + "其中第%s部分是：%s<n>" % (index + 1, part.replace("\n", "<n>"))
        sft_prompt = sft_prompt + "根据上面内容回答问题：<n>%s" % query.replace("\n", "<n>")
        query_ans = self.glm_inference(sft_prompt.replace("\n", "<n>"))
        return query_ans

    def cut_doc(self, data, piece_len=750, single_piece_max_len=1700):
        if len(data)<single_piece_max_len:
            return [data]
        piece_data = []
        index = 0
        while index < len(data):
            piece_data.append(data[index: index + piece_len])
            index += piece_len
        return piece_data
