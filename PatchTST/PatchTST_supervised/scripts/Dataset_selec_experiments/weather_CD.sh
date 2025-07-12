#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_weather
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Linear_weather.out

module load devel/cuda/11.8

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=Linear_CD

for data_lag in 0 48 96 336
do
    for pred_len in 96 720
    do
        python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path _weather_shower_lag_$data_lag.csv \
        --target T_degC_LAG \
        --model_id _weather_shower_lag_$data_lag'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 2 \
        --des 'Exp' \
        --itr 1 --batch_size 16 --patience 10 >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'$pred_len.log 
    done
done