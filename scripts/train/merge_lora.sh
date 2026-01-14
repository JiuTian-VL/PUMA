#!/bin/bash

MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /model/stage1_distill \
    --model-base $MODEL_NAME  \
    --save-model-path /model/stage1_merged \
    --safe-serialization