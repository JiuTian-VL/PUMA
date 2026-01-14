#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


MODEL_NAME="/model/stage1_merged"

# 4*80G GPU
deepspeed --include localhost:0,1,2,3 --master_port 25671 src/training/train_pre.py \
    --lora_enable True \
    --lora_namespan_exclude "['embed_tokens', 'router']" \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /data/M-BEIR/query/union_train/mbeir_union_up_train.jsonl \
    --image_folder /data/M-BEIR/ \
    --cand_path /data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_folder /data/M-BEIR/ \
    --inst_path /data/M-BEIR/instructions/query_instructions.tsv \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /model/stage2_lora \
    --num_train_epochs 2 \
    --per_device_train_batch_size 150 \
    --grad_cache_micro_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --min_pixels $((4 * 28 * 28)) \
    --max_pixels $((300 * 28 * 28)) \
    --learning_rate 3e-4 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --scale 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 3000 \
    --decay_rate 0.2 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --softmax_temperature 0.03 \
    --grad_cache_enable False \
    --training_stage v2 \
    --compression False \
    --info_hard False \
    --drop False \
    --hard_norm False \
    --router False \
    --layer_num 12
