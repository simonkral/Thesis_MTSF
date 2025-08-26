#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=ModernTCN_national_illness


# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=104
model_name=ModernTCN



for random_seed in 2021 2022 2023 2024 2025
do
    for pred_len in 24 36 48 60
    do
        for channel_handling in CI_loc CI_glob CD Delta
        do
            python -u run_longExp.py \
                --random_seed $random_seed \
                --is_training 1 \
                --model_id national_illness_$seq_len'_'$pred_len \
                --model $model_name \
                --root_path ./dataset/ \
                --data_path national_illness.csv \
                --data custom \
                --features M \
                --seq_len $seq_len \
                --pred_len $pred_len \
                --label_len 0 \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 51 \
                --small_size 5 \
                --dims 64 64 64 64 \
                --head_dropout 0.0 \
                --enc_in 7 \
                --dropout 0.1 \
                --itr 1 \
                --train_epochs 100 \
                --batch_size 32 \
                --patience 5 \
                --learning_rate 0.0025 \
                --des Exp \
                --lradj constant \
                --use_multi_scale 0 \
                --small_kernel_merged 0 \
                --channel_handling $channel_handling \
                --delta_factor 0.5 \
                >logs/LongForecasting/$model_name'_'exchange_$seq_len'_'$pred_len.log
        done
    done
done