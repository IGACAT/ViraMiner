#!/usr/bin/env bash
# --input_file $1
# --model_path $2
# --output_path $3
# GPU id $4
export CUDA_VISIBLE_DEVICES=$4
python ~/Applications/ViraMiner/predict_only.py --input_file $1 --model_path $2 --output_path $3 &> $3/predict.vm.out
