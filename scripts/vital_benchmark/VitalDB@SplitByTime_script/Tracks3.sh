export CUDA_VISIBLE_DEVICES=0

seq_len=30

root_path_name=./dataset/VitalDB/data/data_every_1m
data_path_name='*.csv'
model_id_name=VitalDB@MaskedTracks+DG
data_name=folder

model_name=S4
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 1 --masked_training \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 1 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.001
done

model_name=PatchTST
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 1 --masked_training \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 1 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.0005
done

model_name=iTransformer
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 1 --masked_training \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 4 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.0005
done

model_name=FourierGNN
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 1 --masked_training \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 4 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.001
done


model_name=DLinear
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 1 --masked_training \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.0005
done


model_name=MambaTS
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 1 --masked_training \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 4 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --patch_len 5 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.001 --mamba_mode 0 --shuffle_mode 5
done


model_name=S4
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 0 --masked_training --domain_to_generalize 2 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 1 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.001
done

model_name=PatchTST
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 0 --masked_training --domain_to_generalize 2 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 1 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.0005
done

model_name=iTransformer
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 0 --masked_training --domain_to_generalize 2 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 4 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.0005
done

model_name=FourierGNN
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 0 --masked_training --domain_to_generalize 2 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 4 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.001
done

model_name=FourierGNN
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 0 --masked_training --domain_to_generalize 2 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 256 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.001
done


model_name=DLinear
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 0 --masked_training --domain_to_generalize 2 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --patch_len 10 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.0005
done


model_name=MambaTS
for pred_len in 1 3 10 30
do
    python -u run.py \
        --task_name masked_vital_forecast \
        --is_training 0 --masked_training --domain_to_generalize 2 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --norm_pkl ./dataset/norm_pickle/VitalDB-1m@Time.pkl \
        --enc_in 17 \
        --dec_in 17 \
        --c_out 17 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MP --target BIS/BIS --max_num_cases -1 --num_target 8 \
        --seq_len $seq_len \
        --label_len $seq_len \
        --pred_len $pred_len \
        --e_layers 4 \
        --d_layers 2 \
        --factor 1 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 512 \
        --dropout 0.2 --RevIN_mode 0 \
        --patch_len 5 --stride 5 \
        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate 0.001 --mamba_mode 0 --shuffle_mode 0
done
