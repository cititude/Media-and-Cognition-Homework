CUDA_VISIBLE_DEVICES=3 \
    python main.py \
    --model_name sedensenet \
    --device   cuda \
    --fix_seed  \
    --only_test \
    --weights_path ./final/model_299.pt \
    --cam \
    --maxdata 99999