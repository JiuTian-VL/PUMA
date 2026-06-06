#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

BASE_DIR="/data/output"

python -m src.eval.mbeir_retriever \
    --config_path scripts/all/retrieval.yaml \
    --uniir_dir $BASE_DIR/query \
    --mbeir_data_dir $BASE_DIR/cand \
    --enable_retrieval