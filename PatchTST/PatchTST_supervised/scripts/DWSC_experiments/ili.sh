#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=DLinear_ILI
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/DLinear_ILI.out

module load devel/cuda/11.8

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Simon_DWSC

for kernel_size in 3
do
    for n_blocks in 1
    do
        for pred_len in 24 36 48 60
        do
        python -u run_longExp.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path national_illness.csv \
            --model_id national_illness_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len 18 \
            --pred_len $pred_len \
            --kernel_size $kernel_size \
            --n_blocks $n_blocks \
            --enc_in 7 \
            --des 'Exp' \
            --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'$pred_len'_kernel'$kernel_size'_blocks'$n_blocks.log
        done
    done
done
