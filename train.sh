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
  --eval_steps 30 \
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
  --num_train_epochs 4.0 \
  --max_seq_length 512 \
  --evaluation_strategy epoch \
  --output_dir output/rm_model \
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
