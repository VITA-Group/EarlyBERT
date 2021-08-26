#!/bin/bash
# Script to run an EarlyBERT finetuning experiment on SQuAD-v1 dataset.
#
# Modified by:
#     - Xiaohan Chen
#     - xiaohan.chen@utexas.edu
#     - Last modified on: Aug 25, 2021

seed=42
squad_dir=/path/to/your/squad/data
model_type=bert-base-uncased

# Ticket searching hyperparamters
search_lr=3e-5
l1_loss_coef=1e-4

# Ticket drawing hyperparameters
self_pruning_ratio=0.333333
self_pruning_method="layerwise"
inter_pruning_ratio=0.4
inter_pruning_method="global"
slimming_coef_step=0.2

# Efficient training hyperparameters
retrain_lr=3e-5
retrain_num_epochs=3

train_file=$squad_dir/train-v1.1.json
predict_file=$squad_dir/dev-v1.1.json
version_2_with_negative=False

# Process output directory
search_output_dir="logs/earlybert_squad-v1_${model_type}_seed-${seed}_lr-${search_lr}_search_l1-${l1_loss_coef}"
retrain_output_dir="${search_output_dir}_lt-${slimming_coef_step}_self-${self_pruning_method}-pr${self_pruning_ratio}_inter-${inter_pruning_method}-pr${inter_pruning_ratio}_retrain-${retrain_lr}-${retrain_num_epochs}ep"

# Ticket searching stage
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_squad_earlybert_search.py \
    --model_type bert \
    --model_name_or_path $model_type \
    --do_train \
    --do_eval \
    --do_lower_case \
    --save_steps 2500 \
    --train_file $train_file \
    --predict_file $predict_file \
    --version_2_with_negative $version_2_with_negative \
    --per_gpu_train_batch_size 12 \
    --learning_rate $search_lr \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --l1_loss_coef $l1_loss_coef \
    --output_dir $search_output_dir \
    --seed $seed \
    --overwrite_output_dir

# Ticket-drawing and efficient-training stage

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_squad_sa_inter_slimming_lt_pruning.py \
    --model_type bert \
    --model_name_or_path "${search_output_dir}/checkpoint-0" \
    --do_train \
    --do_eval \
    --do_lower_case \
    --save_steps 2500 \
    --train_file $train_file \
    --predict_file $predict_file \
    --version_2_with_negative $version_2_with_negative \
    --per_gpu_train_batch_size 12 \
    --learning_rate $retrain_lr \
    --num_train_epochs $retrain_num_epochs \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $retrain_output_dir \
    --seed $seed \
    --overwrite_output_dir \
    --self_pruning_ratio   $self_pruning_ratio   \
    --self_pruning_method  $self_pruning_method  \
    --inter_pruning_ratio  $inter_pruning_ratio  \
    --inter_pruning_method $inter_pruning_method \
    --slimming_coef_step   $slimming_coef_step   \
    --self_slimming_coef_file  "${search_output_dir}/self_slimming_coef_records.npy"  \
    --inter_slimming_coef_file "${search_output_dir}/inter_slimming_coef_records.npy"

