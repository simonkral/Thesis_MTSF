#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_final_ILI
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Linear_final.out

module load devel/cuda/11.8

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Linear_final

### CI local and global ### 
for channel_handling in CI_loc CI_glob
do
    for pred_len in 24 60
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
            --channel_handling $channel_handling \
            --enc_in 7 \
            --des 'Exp' \
            --itr 1 --batch_size 32 --patience 10 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'$pred_len.log
    done
done

### CD and Delta for different cd_weight_decay ### 
for channel_handling in CD Delta
do
    for cd_weight_decay in 0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1
    do
        for pred_len in 24 60
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
                --cd_weight_decay $cd_weight_decay \
                --channel_handling $channel_handling \
                --enc_in 7 \
                --des 'Exp' \
                --itr 1 --batch_size 32 --patience 10 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'$pred_len.log
        done
    done
done