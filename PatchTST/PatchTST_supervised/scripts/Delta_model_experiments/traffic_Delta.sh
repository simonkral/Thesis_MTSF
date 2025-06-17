#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_Delta_traffic
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Linear_Delta_traffic.out

module load devel/cuda/11.8

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Linear_Delta

for cd_regularization in 0 0.25 0.5 0.75 1
do
    for pred_len in 96
    do
        python -u run_longExp.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path traffic.csv \
            --model_id traffic_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --cd_regularization $cd_regularization \
            --enc_in 862 \
            --des 'Exp' \
            --itr 1 --batch_size 4 --learning_rate 0.05 --use_amp >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'$pred_len'_cd_reg'$cd_regularization.log
    done
done

# reduced --batch_size from 16 to 8 due to: torch.cuda.OutOfMemoryError: CUDA out of memory
# down to 4
# --use_amp added for mixed precision training