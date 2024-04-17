#!/bin/bash
cd ../../
loss_fn=(l2 l1 all)

for loss in "${loss_fn[@]}"; do
    python run.py \
        --overrides group_name=attention_titanet_filter_every \
        --overrides exp_name=filter_every_ablation \
        --overrides trainer.loss=${loss}\
        --overrides trainer.filter_every=true \
        --overrides seed=1
done

for loss in "${loss_fn[@]}"; do
    python run.py \
        --overrides group_name=attention_titanet_filter_every \
        --overrides exp_name=filter_every_ablation \
        --overrides trainer.loss=${loss}\
        --overrides trainer.filter_every=true \
        --overrides seed=1
done