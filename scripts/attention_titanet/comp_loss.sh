#!/bin/bash
cd ../../
loss_fn=(all distill l2 l1)

for seed in {1..3}; do
    for loss in "${loss_fn[@]}"; do
        python run.py \
            --configs= \
            --overrides group_name=attention_titanet \
            --overrides exp_name=comp_loss_${loss} \
            --overrides trainer.loss=${loss}\
            --overrides seed="$seed"
    done
done