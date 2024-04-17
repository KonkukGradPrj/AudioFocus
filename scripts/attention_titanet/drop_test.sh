#!/bin/bash
cd ../../
loss_fn=l2

python run.py \
    --overrides group_name=attention_titanet_filter_every \
    --overrides exp_name=drop_out \
    --overrides trainer.loss=${loss_fn}\
    --overrides trainer.filter_every=true \
    --overrides seed=1