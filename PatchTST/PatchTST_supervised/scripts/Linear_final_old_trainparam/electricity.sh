#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_Final_electricity
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Linear_Final_electricity.out

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

### CI local and global ### 
for channel_handling in CI_loc CI_glob
do
    for pred_len in 96
    do
        python -u run_longExp.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path electricity.csv \
            --model_id Electricity_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --channel_handling $channel_handling \
            --enc_in 321 \
            --des 'Exp' \
            --itr 1 --batch_size 16 --patience 10 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Etth2_$seq_len'_'$pred_len.log
    done
done
