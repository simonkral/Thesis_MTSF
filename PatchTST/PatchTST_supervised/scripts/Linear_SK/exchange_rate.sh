#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_Final_exchange
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Linear_Final_exchange.out

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
model_name=Linear_final

batch_size=32
patience=20
learning_rate=0.005


for random_seed in 2021 2022 2023 2024 2025
do
    ### CI local and global ### 
    for channel_handling in CI_loc CI_glob
    do
        for pred_len in 96 192 336 720
        do
            python -u run_longExp.py \
                --random_seed $random_seed \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path exchange_rate.csv \
                --model_id Exchange_$seq_len'_'$pred_len \
                --model $model_name \
                --data custom \
                --features M \
                --seq_len $seq_len \
                --pred_len $pred_len \
                --channel_handling $channel_handling \
                --enc_in 8 \
                --des 'Exp' \
                --itr 1 --batch_size $batch_size --patience $patience --learning_rate $learning_rate \
                >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'$pred_len.log
        done
    done

    ### CD and Delta for different cd_weight_decay ### 
    for channel_handling in CD Delta
    do
        for cd_weight_decay in 0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e-0 10 100
        do 
            for pred_len in 96 192 336 720
            do
                python -u run_longExp.py \
                    --random_seed $random_seed \
                    --is_training 1 \
                    --root_path ./dataset/ \
                    --data_path exchange_rate.csv \
                    --model_id Exchange_$seq_len'_'$pred_len \
                    --model $model_name \
                    --data custom \
                    --features M \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --cd_weight_decay $cd_weight_decay \
                    --channel_handling $channel_handling \
                    --enc_in 8 \
                    --des 'Exp' \
                    --skip_1st_epoch 1 \
                    --itr 1 --batch_size $batch_size --patience $patience --learning_rate $learning_rate \
                    >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'$pred_len.log
            done
        done
    done
done