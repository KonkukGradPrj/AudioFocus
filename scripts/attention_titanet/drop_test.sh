#!/bin/bash
cd ../../
loss_fn=cross

python run.py \
    --overrides group_name=attention_titanet \
    --overrides exp_name=drop_out \
    --overrides trainer.loss=${loss_fn}\
    --overrides seed=1