#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Linear_CD_ILI
#SBATCH --output=/pfs/work9/workspace/scratch/ma_skral-SK_thesis_2025/Thesis_MTSF/slurm/Linear_ILI.out

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

seq_len=104
model_name=Linear_CD

for data_lag in 0 9 18 24 60
do
    for pred_len in 24
    do
        python -u run_longExp.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path _national_illness_shower_lag_$data_lag.csv \
            --target %_WEIGHTED_ILI_LAG \
            --model_id national_illness_shower_lag_$data_lag_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len 18 \
            --pred_len $pred_len \
            --enc_in 2 \
            --des 'Exp' \
            --itr 1 --batch_size 32 --patience 10 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'$pred_len.log
    done
done