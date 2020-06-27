CUDA_VISIBLE_DEVICES=1 \
    python main.py \
    --model_name sedensenet \
    --record_dir record_dir   \
    --record_file sedensenet_result.txt \
    --print_freq 20  \
    --device   cuda \
    --fix_seed  \
    --maxdata 2 \
    --data_path  ../image_exp/Classification/Data