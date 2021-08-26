#!/bin/bash
# Script to run an EarlyBERT finetuning experiment on one task in GLUE benchmark.
#
# Please modify the value of variable `glue_dir` to location of the GLUE dataset.
# You can change the value of variable `task_name` to run experiments in other
#   tasks in GLUE benchmark.
#
# Modified by:
#     - Xiaohan Chen
#     - xiaohan.chen@utexas.edu
#     - Last modified on: Aug 25, 2021

seed=42
task_name=QNLI
glue_dir=/path/to/your/glue/data
model_type=bert-base-cased
per_device_batch_size=32

# Ticket searching hyperparamters
search_lr=2e-5
l1_loss_self_coef=1e-4
l1_loss_inter_coef=1e-4

# Ticket drawing hyperparameters
self_pruning_ratio=0.333333
self_pruning_method="layerwise"
inter_pruning_ratio=0.4
inter_pruning_method="global"
slimming_coef_step=0.2

# Efficient training hyperparameters
retrain_lr=4e-5
retrain_num_epochs=2

# Process output directory
search_output_dir="logs/${task_name}_${model_type}_seed-${seed}_bs-${per_device_batch_size}x4_lr-${search_lr}_search-sa-${l1_loss_self_coef}-inter-${l1_loss_inter_coef}"
retrain_output_dir="${search_output_dir}_lt-${slimming_coef_step}_self-${self_pruning_method}-pr${self_pruning_ratio}_inter-${inter_pruning_method}-pr${inter_pruning_ratio}_retrain-${retrain_lr}-${retrain_num_epochs}ep/"

python run_glue_sa_inter_slimming.py \
    --model_name_or_path $model_type \
    --task_name $task_name \
    --do_train \
    --do_eval \
    --save_steps 2500 \
    --data_dir $glue_dir/$task_name \
    --max_seq_length 128 \
    --per_device_train_batch_size $per_device_batch_size \
    --learning_rate $search_lr \
    --num_train_epochs 3.0 \
    --output_dir $slimming_output_dir \
    --seed $seed \
    --overwrite_output_dir \
    --l1_loss_self_coef  $l1_loss_self_coef \
    --l1_loss_inter_coef $l1_loss_inter_coef \
    --max_epochs 1

cp "${search_output_dir}/vocab.txt" "${search_output_dir}/checkpoint-0"

python run_glue_sa_inter_slimming_lt_pruning.py \
    --model_name_or_path "${search_output_dir}/checkpoint-0" \
    --task_name $task_name \
    --logging_steps 1000 \
    --do_train \
    --do_eval \
    --save_steps 2500 \
    --logging_steps $logging_steps \
    --evaluate_during_training \
    --data_dir $glue_dir/$task_name \
    --max_seq_length 128 \
    --per_device_train_batch_size $per_device_batch_size \
    --learning_rate $retrain_lr \
    --num_train_epochs $retrain_num_epochs \
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

