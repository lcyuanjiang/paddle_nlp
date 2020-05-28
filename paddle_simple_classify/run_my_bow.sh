python -u run_classify_blog.py \
       --use_cuda False \
       --data_dir data \
       --epoch 10 \
       --random_seed 42 \
       --max_seq_len 64 \
       --num_labels 3 \
       --do_train True \
       --do_val True \
       --do_infer False \
       --batch_size 16 \
       --checkpoints checkpoints \
       --save_steps 100 \
       --vocab_size 240428 \
       --lr 0.002 \
       --verbose True \
       --vocab_path data/vocab.txt \
       --skip_steps 10\
       --validation_steps 50 \
       --save_steps 50 
