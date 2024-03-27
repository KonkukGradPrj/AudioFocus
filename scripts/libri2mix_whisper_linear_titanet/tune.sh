#!/bin/bash
cd ../../

# Define the input and label ratios
weight_decay=(0.001 0.0001 0.00001)
learning_rate=(0.01 0.001 0.0001 0.00001)


for seed in {1..5}; do
    for lr in "${learning_rate[@]}"; do
        for wd in "${weight_decay[@]}"; do
            python run.py \
                --overrides group_name=tune_libri2mix_linear_titanet \
                --overrides exp_name=lr"$lr"_wd"$wd" \
                --overrides trainer.optimizer.lr="$lr" \
                --overrides trainer.optimizer.weight_decay="$wd" \
                --overrides seed="$seed"
        done
    done
done