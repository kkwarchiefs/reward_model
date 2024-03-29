CUDA_VISIBLE_DEVICES=0  python3 trans_classify.py \
  --model_name_or_path /search/ai/pretrain_models/models--nghuyong--ernie-3.0-base-zh \
  --do_train \
  --do_eval \
  --do_predict \
  --pred_path datas/query_intention/dev.tsv \
  --train_path datas/query_intention/train.tsv \
  --dev_path datas/query_intention/dev.tsv \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 16.0 \
  --max_seq_length 32 \
  --evaluation_strategy epoch \
  --output_dir output/tmp \
  --overwrite_output_dir \
  --save_steps -1


python3 -m torch.distributed.launch --nproc_per_node 4  trans_classify_glm.py \
  --model_name_or_path /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --do_predict \
  --pred_path datas/resp_format/test.tsv \
  --train_path datas/resp_format/train.tsv \
  --dev_path datas/resp_format/dev.tsv \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/resp_format \
  --overwrite_output_dir \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 2  trans_classify_glm.py \
  --model_name_or_path /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --do_predict \
  --pred_path datas/resp_second/dev.tsv \
  --train_path datas/resp_second/train.tsv \
  --dev_path datas/resp_second/dev.tsv \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 600 \
  --evaluation_strategy epoch \
  --output_dir output/resp_format_sec \
  --overwrite_output_dir \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 4 trans_classify_cpt.py \
  --model_name_or_path /search/ai/pretrain_models/cpt-large/ \
  --do_train \
  --do_eval \
  --do_predict \
  --pred_path datas/resp_second/dev.tsv \
  --train_path datas/resp_second/train.tsv \
  --dev_path datas/resp_second/dev.tsv \
  --per_device_train_batch_size 3 \
  --learning_rate 2e-6 \
  --num_train_epochs 3.0 \
  --max_seq_length 1024 \
  --evaluation_strategy epoch \
  --output_dir output/resp_format_4th \
  --overwrite_output_dir \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 4 trans_classify_cpt.py \
  --model_name_or_path /search/ai/pretrain_models/cpt-large/ \
  --do_train \
  --do_eval \
  --do_predict \
  --pred_path datas/resp_0324/dev.tsv \
  --train_path datas/resp_0324/train.tsv \
  --dev_path datas/resp_0324/dev.tsv \
  --per_device_train_batch_size 3 \
  --learning_rate 2e-6 \
  --logging_steps 50 \
  --num_train_epochs 3.0 \
  --max_seq_length 1024 \
  --evaluation_strategy epoch \
  --output_dir output/resp_format_0324 \
  --overwrite_output_dir \
  --save_steps -1

 python3 -m torch.distributed.launch --nproc_per_node 8 trans_classify_cpt.py \
  --model_name_or_path /search/ai/pretrain_models/cpt-large/ \
  --do_train \
  --do_eval \
  --do_predict \
  --pred_path datas/resp_0324/dev.tsv \
  --train_path datas/resp_0324/train.tsv \
  --dev_path datas/resp_0324/dev.tsv \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-6 \
  --logging_steps 50 \
  --num_train_epochs 6.0 \
  --max_seq_length 1024 \
  --evaluation_strategy epoch \
  --output_dir output/resp_format_0326 \
  --overwrite_output_dir \
  --save_steps -1


python3 -m torch.distributed.launch --nproc_per_node 1 trans_classify_cpt.py \
  --model_name_or_path output/resp_format_0324 \
  --do_predict \
  --pred_path datas/resp_0324/dev.tsv \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 3 \
  --output_dir output/multi_tasks3 \
  --max_seq_length 1024 \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 4 trans_classify_glm.py \
  --model_name_or_path output/resp_format \
  --do_predict \
  --pred_path datas/resp_format/pred.tsv \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --output_dir output/multi_tasks2 \
  --max_seq_length 512 \
  --save_steps -1



CUDA_VISIBLE_DEVICES=6  python3 reward_rank.py \
  --model_name_or_path /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --logging_steps 20 \
  --train_group_size 2 \
  --learning_rate 1e-5 \
  --num_train_epochs 6.0 \
  --max_seq_length 512 \
  --evaluation_strategy steps \
  --eval_steps 30 \
  --output_dir output/rm_model \
  --overwrite_output_dir \
  --save_steps -1

deepspeed --num_gpus=4 reward_rank.py \
  --model_name_or_path /search/ai/kaitongyang/RLHF_DEBUG/RM/glm_0.5 \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --logging_steps 10 \
  --train_group_size 3 \
  --learning_rate 1e-7 \
  --num_train_epochs 6.0 \
  --max_seq_length 512 \
  --evaluation_strategy steps \
  --output_dir output/rm_model \
  --overwrite_output_dir \
  --save_steps -1 \
  --bf16 True \
  --weight_decay 0.01\
  --adam_beta2 0.95 \
  --deepspeed config_blocklm_10B_cnndm.json

python3 -m torch.distributed.launch --nproc_per_node 4 reward_rank.py \
  --model_name_or_path /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 8 \
  --logging_steps 20 \
  --train_group_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/rm_model \
  --overwrite_output_dir \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 8 reward_rank.py \
  --model_name_or_path /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --logging_steps 20 \
  --train_group_size 3 \
  --rank_list_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/rm_model_short \
  --overwrite_output_dir \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 8 reward_rank_random.py \
  --model_name_or_path /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --logging_steps 20 \
  --train_group_size 3 \
  --rank_list_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/rm_model_short \
  --overwrite_output_dir \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 8 reward_rank_pooling.py \
  --model_name_or_path  /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --logging_steps 20 \
  --train_group_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/rm_model \
  --overwrite_output_dir \
  --save_steps -1



python3 -m torch.distributed.launch --nproc_per_node 8 reward_rank.py \
  --model_name_or_path  /search/ai/pretrain_models/chinese-roberta-wwm-ext-large/ \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --logging_steps 20 \
  --train_group_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/rm_model \
  --overwrite_output_dir \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 8 reward_rank.py \
  --model_name_or_path  /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --logging_steps 20 \
  --train_group_size 6 \
  --rank_list_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/rm_model \
  --overwrite_output_dir \
  --save_steps -1

python3 -m torch.distributed.launch --nproc_per_node 4 reward_rank.py \
  --model_name_or_path  /search/ai/pretrain_models/glm-large-chinese/ \
  --do_train \
  --do_eval \
  --train_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/train.tsv \
  --dev_path /search/ai/jamsluo/GLM_RLHF/reward_model/reward_data/dev.tsv \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --logging_steps 20 \
  --train_group_size 3 \
  --rank_list_size 2 \
  --learning_rate 1e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/rm_model \
  --overwrite_output_dir \
  --save_steps -1

CUDA_VISIBLE_DEVICES=0  python3 trans_classify.py \
  --model_name_or_path bert-base-chinese \
  --do_train \
  --do_eval \
  --do_predict \
  --pred_path datas/query_classify/dev.tsv \
  --train_path datas/query_classify/train.tsv \
  --dev_path datas/query_classify/dev.tsv \
  --per_device_train_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 32 \
  --evaluation_strategy epoch \
  --output_dir output/bert_baes \
  --overwrite_output_dir \
  --save_steps -1


CUDA_VISIBLE_DEVICES=0  python3 trans_classify.py \
  --model_name_or_path output/multi_tasks \
  --do_predict \
  --pred_path datas/query_tasks/train.tsv \
  --per_device_train_batch_size 4 \
  --output_dir output/multi_tasks2 \
  --max_seq_length 32 \
  --save_steps -1




CUDA_VISIBLE_DEVICES=0  python3 trans_classify_pipeline.py \
  --model_name_or_path output/multi_tasks \
  --per_device_train_batch_size 4 \
  --output_dir output/multi_tasks2 \
  --max_seq_length 32 \
  --save_steps -1

CUDA_VISIBLE_DEVICES=0  python3 rewar.py \
  --model_name_or_path /search/ai/pretrain_models/models--nghuyong--ernie-3.0-base-zh \
  --do_train \
  --do_eval \
  --do_predict \
  --pred_path datas/query_intention/dev.tsv \
  --train_path datas/query_intention/train.tsv \
  --dev_path datas/query_intention/dev.tsv \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 16.0 \
  --max_seq_length 32 \
  --evaluation_strategy epoch \
  --output_dir output/tmp \
  --overwrite_output_dir \
  --save_steps -1


python3 -m torch.distributed.launch --nproc_per_node 8 multi_qa.py \
  --model_name_or_path /search/ai/pretrain_models/infoxlm-base/ \
  --dataset_name squad.py \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 80 \
  --logging_steps 20 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384  \
  --doc_stride 128 \
  --overwrite_output_dir \
  --version_2_with_negative \
  --evaluation_strategy epoch \
  --output_dir /search/ai/jamsluo/GLM_RLHF/reward_model/output/multi_qa_v2


python3 -m torch.distributed.launch --nproc_per_node 8 multi_qa.py \
  --model_name_or_path /search/ai/pretrain_models/infoxlm-base/ \
  --dataset_name squad.py \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 100 \
  --logging_steps 20 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 512  \
  --doc_stride 256 \
  --overwrite_output_dir \
  --version_2_with_negative \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --output_dir /search/ai/jamsluo/GLM_RLHF/reward_model/output/squad_cmrc_compare




python3 -m torch.distributed.launch --nproc_per_node 8 multi_qa.py \
  --model_name_or_path /search/ai/pretrain_models/infoxlm-base/ \
  --dataset_name squad_zh_en.py \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 80 \
  --logging_steps 20 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 512  \
  --doc_stride 256 \
  --overwrite_output_dir \
  --version_2_with_negative \
  --evaluation_strategy epoch \
  --output_dir /search/ai/jamsluo/GLM_RLHF/reward_model/output/squad_cmrc_du_wiki_nyt


torchrun --nnodes 1 --nproc_per_node 4 multi_qa.py \
  --model_name_or_path /search/ai/pretrain_models/infoxlm-base/ \
  --dataset_name squad_rel.py \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 50 \
  --logging_steps 20 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --lr_scheduler_type cosine \
  --max_seq_length 512  \
  --doc_stride 128 \
  --seed 66 \
  --overwrite_cache True \
  --max_answer_length 128 \
  --overwrite_output_dir \
  --version_2_with_negative \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --output_dir /search/ai/jamsluo/GLM_RLHF/reward_model/output/suqad_toolgpt_rel_multi_lang_human_corr


torchrun --nnodes 1 --nproc_per_node 8 multi_qa.py \
  --model_name_or_path /search/ai/pretrain_models/infoxlm-base/ \
  --dataset_name squad_rel_en.py \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 70 \
  --logging_steps 20 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512  \
  --doc_stride 128 \
  --overwrite_cache True \
  --max_answer_length 128 \
  --overwrite_output_dir \
  --version_2_with_negative \
  --evaluation_strategy epoch \
  --output_dir /search/ai/jamsluo/GLM_RLHF/reward_model/output/suqad_toolgpt_rel_English


CUDA_VISIBLE_DEVICES=0  python3 multi_qa.py \
  --model_name_or_path /search/ai/jamsluo/GLM_RLHF/reward_model/output/suqad_toolgpt_rel \
  --dataset_name squad_rel.py \
  --do_predict \
  --logging_steps 20 \
  --max_seq_length 512  \
  --doc_stride 128 \
  --overwrite_cache True \
  --max_answer_length 128 \
  --version_2_with_negative \
  --output_dir /search/ai/jamsluo/GLM_RLHF/reward_model/output/suqad_toolgpt_rel


python3 multi_qa.py \
  --model_name_or_path /search/ai/pretrain_models/infoxlm-base/ \
  --dataset_name squad_rel.py \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 40 \
  --logging_steps 20 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 512  \
  --doc_stride 128 \
  --max_answer_length 128 \
  --overwrite_output_dir \
  --version_2_with_negative \
  --evaluation_strategy epoch \
  --output_dir /search/ai/jamsluo/GLM_RLHF/reward_model/output/suqad_toolgpt_rel
