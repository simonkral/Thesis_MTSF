#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Simon_DWSC_ETTh1
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Simon_DWSC.out

module load devel/cuda/11.8

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Simon_DWSC

for conv_kernel_size in 3 5 9 13 17 25 33
do
    for n_blocks in 1
    do
        for pred_len in 96 720
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
            --conv_kernel_size $conv_kernel_size \
            --n_blocks $n_blocks \
            --enc_in 7 \
            --des 'Exp' \
            --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'$pred_len'_kernel'$conv_kernel_size'_blocks'$n_blocks.log
        done
    done
done