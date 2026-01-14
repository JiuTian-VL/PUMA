#!/bin/bash


SAVEPATH="/data/output"

DATASET_LIST=(
    "mbeir_visualnews_task0"
    "mbeir_mscoco_task0"
    "mbeir_fashion200k_task0"
    "mbeir_webqa_task1"
    "mbeir_edis_task2"
    "mbeir_webqa_task2"
    "mbeir_visualnews_task3"
    "mbeir_mscoco_task3"
    "mbeir_fashion200k_task3"
    "mbeir_nights_task4"
    "mbeir_oven_task6"
    "mbeir_infoseek_task6"
    "mbeir_fashioniq_task7"
    "mbeir_cirr_task7"
    "mbeir_oven_task8"
    "mbeir_infoseek_task8"
)
export TOKENIZERS_PARALLELISM=false

IDX=1
CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch --multi_gpu --main_process_port 29509 src/eval/infer_eval_multi.py \
    --model-base /model/stage1_merged \
    --model-path /model/stage2_lora \
    --query-path /data/M-BEIR/query/test/${DATASET_LIST[IDX]}_test.jsonl \
    --inst-path /data/M-BEIR/instructions/query_instructions.tsv \
    --cand-path /data/M-BEIR/cand_pool/local/${DATASET_LIST[IDX]}_cand_pool.jsonl \
    --query-cand-path /data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
    --image-folder /data/M-BEIR/ \
    --batch-size 64 \
    --device cuda \
    --max-new-tokens 1 \
    --save-path $SAVEPATH \
    --answer-file $SAVEPATH/${CANDFILE} \
    --layer-num 12



