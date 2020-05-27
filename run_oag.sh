cd main

CUDA_VISIBLE_DEVICES=1 python train.py \
        -run_name oag \
        -data_root /home/fengmingquan/data/oag \
        -embedding_size 32 \
        -n_layer 2 \
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
        -num_epochs 3 \
        -num_batches_per_valid 300 \
        -lr_decay_patience 900
