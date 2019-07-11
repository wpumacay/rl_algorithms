#!/bin/bash

# 1) experiment with different batch sizes, and other hyperparameters as default
batch_sizes=(64 128 256)

echo "Testing over hyperparameter: batch sizes"

for batch_size in "${batch_sizes[@]}"
do
	echo "Running experiment> batch_size=${batch_size} sessionId=sess_batch_size_${batch_size}"
        python ddpg_pytorch_reacher.py --hp_batch_size="${batch_size}" --sessionId="sess_batch_size_${batch_size}"
    done
done

# 2) experiment with different training update frequencies
train_num_updates=(2 5 10)

echo "Testing over hyperparameter: batch sizes"

for tnumup in "${train_num_updates[@]}"
do
	echo "Running experiment> batch_size=${batch_size} sessionId=sess_train_num_updates_${tnumup}"
        python ddpg_pytorch_reacher.py --hp_train_num_updates="${tnumup}" --sessionId="sess_train_num_updates_${tnumup}"
    done
done