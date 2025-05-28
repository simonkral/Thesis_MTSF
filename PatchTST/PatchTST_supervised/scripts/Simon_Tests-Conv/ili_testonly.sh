


module load devel/cuda/11.8



# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Simon_Conv

python -u run_longExp.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 24 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'24.log \
  --checkpoints ./checkpoints/national_illness_104_24_Simon_Conv_custom_ftM_sl104_ll18_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth