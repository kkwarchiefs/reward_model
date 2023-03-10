import json
import pandas as pd
import random





def load_dataset(data_path, datase_type, top_k=4, sample=4):
    ins = pd.read_csv(data_path)
    result = []
    count = 0
    for v in ins.values:
        prompt = v[1]
        response_list = json.loads(v[3])
        response_dict = dict()
        for candidate in response_list:
            response_dict[int(candidate['level'])] = [candidate['score'], candidate['name']]
        response_dict = sorted(response_dict.items(), key = lambda kv:kv[0], reverse=True)
        temp_response_dict = dict()
        min_score = 20000000 
        for i in response_dict:
            level, response = i
            if level in temp_response_dict:
                continue
            else:
                if len(temp_response_dict) > 0:
                    if response[0] >= min_score:
                        continue
                    else:
                        min_score = response[0]
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
                    temp_result.append(temp_response_dict[i][1])
                temp_index = temp_index + 1
            result.append([prompt,temp_result])       
        '''
        all_index = list(range(len(temp_response_dict)))
        candidate_index = all_index[:int(top_k//2)] + all_index[-(top_k-int(top_k//2)):]
        print(candidate_index)
        result = []
        temp_index = 0
        for i in temp_response_dict:
            if temp_index in candidate_index:
                result.append(temp_response_dict[i])
            temp_index = temp_index + 1
        print(result)
        break 
        ''' 
        count = count + 1
        
    print(len(result)) 

    see = 0
    print(result[0][0])
    for i in result[0][1]:
        print("="*10)
        print(i)
       
def load_dataset_pair(data_path, datase_type, top_k=4, sample=4):
    ins = pd.read_csv(data_path)
    result = []
    count = 0
    for v in ins.values:
        prompt = v[1]
        response_list = json.loads(v[3])
        response_dict = dict()
        for candidate in response_list:
            level = int(candidate['level'])
            if level not in response_dict:
                response_dict[level] = [candidate['name']]
            else:
                response_dict[level] = response_dict[level] + [candidate['name']]
        response_dict = sorted(response_dict.items(), key = lambda kv:kv[0], reverse=True)
        if len(response_dict) < 2:
            continue
        best_list = response_dict[0]
        bad_list = response_dict[-1]
        for cur_best in best_list[1]:
            for cur_bad in bad_list[1]:
                result.append([prompt, [cur_best, cur_bad]])
    print(len(result)) 

    see = 700
    print(result[see][0])
    for i in result[see][1]:
        print("="*10)
        print(i)
 
data_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/data/reward_data/reward_model_data_sql_result_20230310111430.csv"
datase_type = "train"
load_dataset_pair(data_path, datase_type)



'''
def process_data():
    root_path = "data/success-0223.json" 
    datas = open(root_path).read().splitlines()
    for data in datas:
        temp = json.loads(data)
        print(temp)
        break


from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/search/ai/pretrain_models/models--nghuyong--ernie-3.0-base-zh")
print(tokenizer(["[SEP]"]))
dd = "我是<n><n>哈哈"

ff = [dd.replace("<n>","##402")]
ff = tokenizer(ff)
print(tokenizer.decode(ff["input_ids"][0]))
'''
