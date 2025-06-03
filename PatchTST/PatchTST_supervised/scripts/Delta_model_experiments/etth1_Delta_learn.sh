#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_Delta_ETTh1
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Linear_Delta.out

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

for pred_len in 96
do
    python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_$seq_len'_'$pred_len \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --learn_cd_regularization True \
        --enc_in 7 \
        --des 'Exp' \
        --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'$pred_len'_cd_reg_learn'.log
done