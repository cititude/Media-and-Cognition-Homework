CUDA_VISIBLE_DEVICES=1 \
    python main.py \
    --model_name sedensenet \
    --device   cuda \
    --fix_seed  \
    --only_test \
    --weights_path ./result/chkpt.pt \
    --data_path ../image_exp/Classification/Data \
    --pred_file result/pred.json