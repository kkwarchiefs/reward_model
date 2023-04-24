import numpy as np
import faiss
import torch
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-ft-5e-6/checkpoint-5000"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-chatglm-6b-ft-1e-5/checkpoint-5000"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-left-chatglm-6b-ft-1e-5/checkpoint-5000"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-1024-chatglm-6b-ft-04-19-19-48/checkpoint-5000"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-1536-chatglm-6b-ft-04-19-20-09/checkpoint-5000"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-left-1536-chatglm-6b-ft-1e-5/checkpoint-5000"
path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/models/chatglm-6b/snapshots/4de8efebc837788ffbfc0a15663de8553da362a2"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-1024-chatglm-6b-ft-04-20-20-31/checkpoint-1000"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-1024-our-chatglm-6b-ft-04-20-18-24/checkpoint-5000"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-1024-chatglm-6b-ft-04-21-14-17/checkpoint-6000"
#path = "/search/ai/pvopliu/chatglm-6b/ChatGLM-6B/ptuning/output/rc-1024-chatglm-6b-ft-04-23-12-05/checkpoint-6000"
device = "cuda:4"
tokenizer_glm  = AutoTokenizer.from_pretrained(path, trust_remote_code=True, truncation_side="left")
#tokenizer_glm  = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
model = model.half().to(device)
model.eval()



#model_name = "embedding_mul_onnx"  # 模型目录名/venus注册模型名称
model_name = "embedding_pooling_onnx"  # 模型目录名/venus注册模型名称
address = "10.212.207.33:8000"  # 机器地址
triton_client = httpclient.InferenceServerClient(url=address)
rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)

class ChatPdf():
    def __init__(self, content_type="web page"):
        self.content_type = content_type
        if self.content_type == "web page":
            self.content_name = "网页"
        elif self.content_type == "PDF file":
            self.content_name = "PDF文件"
        #self.glm_inference = inference_function
    def glm_inference(self, text, is_mrc=False):
        if is_mrc:
            history =  [("中华人民共和国成立于1954年，当毛主席在天安门城墙上宣布中华人民共和国成立了的消息，全国人民都沸腾了。根据上面内容回答问题：<n>中华人民共和国成立于成立于哪一年？",
                    "中华人民共和国成立于1954年。")]
        else:
            history = []
        history = []
        response, history = model.chat(tokenizer_glm, text, history=history, do_sample=False)
        print ("\n\n")
        print (response)
        return response

    def read_data(self, file_path):
        text = open(file_path).read()
        #self.summary = self.get_summary(text)
        #self.questions = self.get_question(text)
        self.doc_piece_list = self.get_doc_embedding_index(text)

    def get_summary(self, data):
        max_len = 1500
        #piece_data = self.cut_doc_plus(data, piece_len=750, single_piece_max_len=max_len, return_only_one=True)
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
                #print ("<n>".join(piece_content).replace("\n", "<n>"))
                summ = self.glm_inference("<n>".join(piece_content).replace("\n", "<n>"))
                final_sum_content.append("第{}部分的主要内容是：".format(i + 1) + summ)
                if len(" ".join(final_sum_content))-20>=max_len:
                    print ("total sum max len")
                    break
        final_sum_content.append("根据上面各个部分内容的介绍生成整个{}的摘要".format(self.content_name))
        #print ("<n>".join(final_sum_content).replace("\n", "<n>"))
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
        #print (input_word.replace("\n", "<n>"))
        questions = self.glm_inference(input_word.replace("\n", "<n>"))
        return questions

    def get_mrc(self, query):
        query_context = self.get_query_context(query)
        #print ("检索上下文：", query_context)
        #sft_prompt = "根据下面内容回答问题：<n>%s<n>" % query.replace("\n", "<n>")
        sft_prompt = "整个%s的摘要是：%s<n>" % (self.content_name, self.summary)
        sft_prompt = ""
        for index, part in enumerate(query_context):
            sft_prompt = sft_prompt + "其中第%s部分是：%s<n>" % (index + 1, part.replace("\n", "<n>"))
        sft_prompt = sft_prompt + "根据上面内容回答问题：<n>%s" % query.replace("\n", "<n>")
        #print (sft_prompt.replace("\n", "<n>"))
        query_ans = self.glm_inference(sft_prompt.replace("\n", "<n>"), is_mrc=True)
        return query_ans


    def get_query_context(self, query, top_k=3):
        query_emb = self.get_query_emb(query)
        # print(query_emb.shape)
        # print(len(self.doc_piece_list))
        # sim_list, index_list = self.doc_piece_emb_index.search(query_emb, min(top_k, len(self.doc_piece_list)))
        scores = np.matmul(query_emb, self.doc_piece_emb_index.transpose(1, 0))[0]
        sortid = np.argsort(-scores)        
        context = []
        #index_list = index_list.tolist()[0]
        #index_list.sort()
        # print(index_list)
        for index in sortid[:3]:
            context.append(self.doc_piece_list[index])
        return context

    def get_embedding(self, doc):
        RM_input = tokenizer(doc, max_length=512, truncation=True, return_tensors="pt", padding=True)
        # print(RM_input)
        RM_batch = [torch.tensor(RM_input["input_ids"]).numpy(), torch.tensor(RM_input["attention_mask"]).numpy()]

        inputs = []
        inputs.append(httpclient.InferInput('input_ids', list(RM_batch[0].shape), 'INT64'))
        inputs.append(httpclient.InferInput('attention_mask', list(RM_batch[1].shape), 'INT64'))
        inputs[0].set_data_from_numpy(RM_batch[0])
        inputs[1].set_data_from_numpy(RM_batch[1])
        output = httpclient.InferRequestedOutput('output')
        # try:
        results = triton_client.infer(
            model_name,
            inputs,
            model_version='1',
            outputs=[output],
            request_id='1'
        )
        results = results.as_numpy('output')
        return results

    def get_query_emb(self, query):
        # embedding = self.openai.Embedding.create(
        #     input=query, model="text-embedding-ada-002"
        # )
        # return np.array([embedding["data"][0]["embedding"]])
        return self.get_embedding(query)

    def get_doc_embedding_index(self, text):

        emb_list = []
        # doc_piece_list = []
        #inputs = self.cut_doc(text, piece_len=500)
        #inputs = self.cut_doc(text, piece_len=500, single_piece_max_len=1500)
        inputs = self.cut_doc_plus(text, piece_len=400, single_piece_max_len=1000)
        # try:
        #     embedding = self.openai.Embedding.create(
        #         input=inputs, model="text-embedding-ada-002"
        #     )
        # except:
        #     return  "", False
        # length = len(embedding["data"])
        # for i in range(length):
        #     doc_piece_list.append(inputs[i])
        #     emb_list.append(embedding["data"][i]["embedding"])
        # emb = np.array(emb_list)
        emb = self.get_embedding(inputs)
        #[5,1356]
        self.doc_piece_emb_index = emb
        #self.doc_piece_emb_index = faiss.IndexFlatL2(emb.shape[1])
        #self.doc_piece_emb_index.add(emb)

        return inputs


    def cut_doc(self, data, piece_len=750, single_piece_max_len=1500):
        if len(data)<single_piece_max_len:
            return [data]
        piece_data = []
        index = 0
        while index < len(data):
            piece_data.append(data[index: index + piece_len])
            index += piece_len
        return piece_data
    def cut_doc_plus(self, data, piece_len=750, single_piece_max_len=1500, return_only_one=False):
        tokens = tokenizer_glm(data)
        tokens_ids = tokens['input_ids'][1:-2]
        if len(tokens_ids) < single_piece_max_len:
            return [data]
        if return_only_one:
            return [tokenizer_glm.decode(tokens_ids[0:single_piece_max_len])]
        index = 0
        piece_data = []
        last_index = 0
        while index < len(tokens_ids):
            index += piece_len
            if index < len(tokens_ids):
                temp_data = tokenizer_glm.decode(tokens_ids[last_index: index])
            else:
                temp_data = tokenizer_glm.decode(tokens_ids[last_index:])
            piece_data.append(temp_data)
            last_index = index
        return piece_data

if __name__=="__main__":
    chatpdf = ChatPdf()
    while True:
        path = input("请输入文件：")
        #path = "./test_datas/test_input.csv"
        if "web" in path:
            chatpdf = ChatPdf()
        else:
            chatpdf = ChatPdf(content_type="PDF file")
        chatpdf.read_data(path)
        while True:
            query = input("\n请输入问题, 输入exit结束：")
            if query=="exit":
                break
            chatpdf.get_mrc(query)
