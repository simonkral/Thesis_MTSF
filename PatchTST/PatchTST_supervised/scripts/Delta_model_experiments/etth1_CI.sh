#!/bin/bash
#SBATCH --time=0:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_ETTh1
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Linear_CI.out

module load devel/cuda/11.8

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

#seq_len=336
seq_len=96
model_name=Linear

for seq_len in 48 96 192 336 720
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
            --enc_in 7 \
            --des 'Exp' \
            --itr 1 --batch_size 32 --patience 10 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'$pred_len.log
    done
done

#for pred_len in 96 192 336 720