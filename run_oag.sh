cd main

CUDA_VISIBLE_DEVICES=3 python train.py \
        -run_name oag \
        -data_root /home/fengmingquan/data/oag \
        -embedding_size 32 \
        -n_layer 1 \
        -n_head 3 \
        -num_bs_instances 3 \
        -num_nested_calls 1 \
        -batch_size 2 \
        -gqa \
        -default_matmul \
        -dense_mat \
        -avg_sample \
        -no_negate \
        -no_sample_neg \
        -skip_trained \
        -no_recursion \
        -patience 20 \
        -num_epochs 50 \
        -num_batches_per_valid 2000 \
        -lr_decay_patience 900
