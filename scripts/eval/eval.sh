#!/bin/bash


export CUDA_VISIBLE_DEVICES=4,5,6,7

BASE_DIR="/data/output"


python -m src.eval.mbeir_retriever \
    --config_path scripts/all/index.yaml \
    --uniir_dir $BASE_DIR/query \
    --mbeir_data_dir $BASE_DIR/cand \
    --enable_create_index

python -m src.eval.mbeir_retriever \
    --config_path scripts/all/ret_single.yaml \
    --uniir_dir $BASE_DIR/query \
    --mbeir_data_dir $BASE_DIR/cand \
    --enable_retrieval