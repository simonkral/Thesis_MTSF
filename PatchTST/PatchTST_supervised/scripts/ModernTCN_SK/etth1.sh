#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=ModernTCN_ETTh1
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/ModernTCN_ETTh1.out

source /pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/miniconda3/etc/profile.d/conda.sh
conda activate PatchTST
module load devel/cuda/11.8

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=ModernTCN

for random_seed in 2021
#for random_seed in 2021 2022 2023 2024 2025
do
    for pred_len in 96 192 336 720
    do
        for channel_handling in CI_loc CI_glob CD Delta
        do
            python -u run_longExp.py \
                --random_seed $random_seed \
                --is_training 1 \
                --model_id ETTh1_$seq_len'_'$pred_len \
                --model $model_name \
                --root_path ./dataset/ \
                --data_path ETTh1.csv \
                --data ETTh1 \
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
                --enc_in 7 \
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
                --small_kernel_merged 0 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'$pred_len.log
        done
    done
done