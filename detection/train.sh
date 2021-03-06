python main.py \
--maxdata 99999 \
--device 2 \
--img_size 1216 \
--epochs 300 \
--batch-size 16 \
--fl_gamma 1.5 \
--data data/traffic.data \
--fix_seed \
--lr 0.001 \
--weights weights/yolov3-spp.weights \
--adam \
--record_dir ./record_dir
