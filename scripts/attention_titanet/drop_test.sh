#!/bin/bash
cd ../../
loss_fn=l2

python run.py \
    --overrides group_name=attention_titanet \
    --overrides exp_name=drop_out \
    --overrides trainer.loss=${loss_fn}\
    --overrides seed=1