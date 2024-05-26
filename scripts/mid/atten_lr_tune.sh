#!/bin/bash
cd ../../



lrs=(0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001)

for lr in "${lrs[@]}"; do
    python run.py \
        --config_name whisper_attention_titanet\
        --overrides group_name=midterm_last_atten_lr_tune \
        --overrides exp_name=lr_${lr}\
        --overrides trainer.filter_every=true \
        --overrides trainer.loss=tri\
        --overrides trainer.optimizer_type=sgd\
        --overrides trainer.epochs=5\
        --overrides trainer.beta=1\
        --overrides trainer.optimizer.lr=${lr}\
        --overrides seed=1
done