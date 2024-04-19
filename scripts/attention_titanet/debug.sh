#!/bin/bash
cd ../../

python run.py \
    --overrides group_name=test \
    --overrides exp_name=test \
    --overrides trainer.loss=l2\
    --overrides trainer.filter_every=true \
    --overrides seed=1
