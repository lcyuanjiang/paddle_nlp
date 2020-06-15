set -eux
python -u run_bin_classify.py \
                   --use_cuda true \
                   --batch_size 8 \
                   --pretrain_model_path ./pretrained_model \
                   --num_class 2 \
                   --train_path ./data/chnsenticorp/train \
                   --dev_path ./data/chnsenticorp/dev \
                   --test_path ./data/chnsenticorp/test \
                   --save_path checkpoints \
                   --num_epoch 3 \
                   --pad_size 128 \
                   --lr 5e-5
