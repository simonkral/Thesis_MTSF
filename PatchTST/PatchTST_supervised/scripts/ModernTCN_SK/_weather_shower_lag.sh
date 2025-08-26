#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=ModernTCN_ETTh1


# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=ModernTCN

for data_lag in 0 48 96 192 336 720
do
    for random_seed in 2021 2022 2023 2024 2025
    do
        for pred_len in 96 192 336 720
        do
            for channel_handling in CI_loc CI_glob CD Delta
            do
                python -u run_longExp.py \
                    --random_seed $random_seed \
                    --is_training 1 \
                    --data_path _weather_shower_lag_$data_lag'_T_degC'.csv \
                    --target T_degC_LAG \
                    --model_id _weather_shower_lag_$data_lag'_T_degC_'$seq_len'_'$pred_len \
                    --model $model_name \
                    --root_path ./dataset/ \
                    --data custom \
                    --features M \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --ffn_ratio 1 \
                    --patch_size 8 \
                    --patch_stride 4 \
                    --num_blocks 1 \
                    --large_size 51 \
                    --small_size 5 \
                    --dims 64 64 64 64 \
                    --head_dropout 0.0 \
                    --enc_in 2 \
                    --dropout 0.3 \
                    --itr 1 \
                    --train_epochs 100 \
                    --batch_size 512 \
                    --patience 20 \
                    --learning_rate 0.0001 \
                    --des Exp \
                    --lradj type3 \
                    --use_multi_scale 0 \
                    --channel_handling $channel_handling \
                    --delta_factor 0.5 \
                    --small_kernel_merged 0 >logs/LongForecasting/$model_name'_'weather_shower_lag_T_degC_$seq_len'_'$pred_len.log
            done
        done
    done
done