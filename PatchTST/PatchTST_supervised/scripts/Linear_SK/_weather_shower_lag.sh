#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_Delta-learn_ETTh1


# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=Linear_final

batch_size=32
patience=20
learning_rate=0.005


for data_lag in 0 48 96 192 336 720
do
    for random_seed in 2021 2022 2023 2024 2025
    do
        for channel_handling in CI_loc CI_glob CD Delta
        do
            for pred_len in 96 192 336 720
            do
                python -u run_longExp.py \
                    --random_seed $random_seed \
                    --is_training 1 \
                    --root_path ./dataset/ \
                    --data_path _weather_shower_lag_$data_lag'_T_degC'.csv \
                    --target T_degC_LAG \
                    --model_id _weather_shower_lag_$data_lag'_T_degC_'$seq_len'_'$pred_len \
                    --model $model_name \
                    --data custom \
                    --features M \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --channel_handling $channel_handling \
                    --enc_in 2 \
                    --des 'Exp' \
                    --itr 1 --batch_size $batch_size --patience $patience --learning_rate $learning_rate \
                    >logs/LongForecasting/$model_name'_'weather_shower_lag_T_degC_$seq_len'_'$pred_len.log
            done
        done
    done
done