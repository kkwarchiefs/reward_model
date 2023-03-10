CHECKPOINT_PATH=/search/ai/kaitongyang/GLM/fintune_model/zhidao_models/GLM-10B-chinese-customization_02-24-00-50/best/mp_rank_00_model_states.pt
MODEL_NAME=glm-10b-chinese
ROOT_OUTPUT_PATH=/search/ai/kaitongyang/GLM_RLHF/RM/pretrain_model/glm_0.3
python3 convert_glm_checkpoint_to_transformers.py $CHECKPOINT_PATH $MODEL_NAME $ROOT_OUTPUT_PATH
