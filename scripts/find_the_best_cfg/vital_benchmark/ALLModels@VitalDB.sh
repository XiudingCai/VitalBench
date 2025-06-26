export CUDA_VISIBLE_DEVICES=0

seq_len=360

root_path_name=./dataset/VitalDB/data/data_every_5s
data_path_name='*.csv'
model_id_name=VitalDB@AllTracks-CFG
data_name=folder

for pred_len in 12 36 120 360
do
    for model_name in S4 PatchTST
    do
        for e_layers in 2 3 4
        do
            for learning_rate in 0.001 0.0005 0.0001
            do
                for d_model in 128 256 512
                do
                    python -u run.py \
                        --task_name masked_vital_forecast \
                        --is_training 1 \
                        --root_path $root_path_name \
                        --data_path $data_path_name \
                        --norm_pkl ./dataset/norm_pickle/VitalDB.pkl \
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
                        --e_layers $e_layers \
                        --d_layers 2 \
                        --factor 1 \
                        --des 'Exp' \
                        --itr 1 \
                        --n_heads 16 \
                        --d_model $d_model \
                        --d_ff 512 \
                        --RevIN_mode 1 \
                        --patch_len 60 --stride 60 \
                        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate $learning_rate --mamba_mode 0 --shuffle_mode 0
                    echo "[CFG] MODEL $model_name, learning_rate $learning_rate, e_layers $e_layers, d_model $d_model"
                done
            done
        done
    done

    for model_name in iTransformer FourierGNN LightTS DLinear MICN
    do
        for e_layers in 2 3 4
        do
            for learning_rate in 0.001 0.0005 0.0001
            do
                for d_model in 128 256 512
                do
                    python -u run.py \
                        --task_name masked_vital_forecast \
                        --is_training 1 \
                        --root_path $root_path_name \
                        --data_path $data_path_name \
                        --norm_pkl ./dataset/norm_pickle/VitalDB.pkl \
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
                        --e_layers $e_layers \
                        --d_layers 2 \
                        --factor 1 \
                        --des 'Exp' \
                        --itr 1 \
                        --n_heads 16 \
                        --d_model $d_model \
                        --d_ff 512 \
                        --RevIN_mode 0 \
                        --patch_len 60 --stride 60 \
                        --train_epochs 10 --patience 3 --batch_size 512 --learning_rate $learning_rate --mamba_mode 0 --shuffle_mode 0
                    echo "[CFG] MODEL $model_name, learning_rate $learning_rate, e_layers $e_layers, d_model $d_model"
                done
            done
        done
    done

    model_name=MambaTS
    for dropout in 0.2 0.3
        do
            for e_layers in 2 3 4
            do
                for learning_rate in 0.001 0.0005 0.0001
                do
                    for d_model in 64 128 256
                    do
                        python -u run.py \
                            --task_name masked_vital_forecast \
                            --is_training 1 \
                            --root_path $root_path_name \
                            --data_path $data_path_name \
                            --norm_pkl ./dataset/norm_pickle/MOVER-SIS.pkl \
                            --enc_in 33 \
                            --dec_in 33 \
                            --c_out 33 \
                            --model_id $model_id_name'_'$seq_len'_'$pred_len \
                            --model $model_name \
                            --data $data_name \
                            --features MP --target MAP_ART --max_num_cases -1 --num_target 13 \
                            --seq_len $seq_len \
                            --label_len $seq_len \
                            --pred_len $pred_len \
                            --e_layers $e_layers \
                            --d_layers 2 \
                            --factor 1 \
                            --des 'Exp' \
                            --itr 1 \
                            --n_heads 16 \
                            --d_model $d_model \
                            --d_ff 512 \
                            --dropout $dropout --RevIN_mode 0 \
                            --patch_len 6 --stride 6 \
                            --train_epochs 10 --patience 3 --batch_size 512 --learning_rate $learning_rate --mamba_mode 0 --shuffle_mode 5
                        echo "[CFG] learning_rate $learning_rate, dropout $dropout, e_layers $e_layers, d_model $d_model"
                    done
                done
            done
        done

done