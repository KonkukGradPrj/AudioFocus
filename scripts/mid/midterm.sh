#!/bin/bash
cd ../../


python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=midterm_v3 \
    --overrides exp_name=attention_titanet\
    --overrides trainer.filter_every=true \
    --overrides trainer.loss=all\
    --overrides trainer.epochs=40\
    --overrides seed=1

python run.py \
    --config_name whisper_linear_titanet\
    --overrides group_name=midterm_v3 \
    --overrides exp_name=linear_titanet\
    --overrides trainer.loss=all\
    --overrides trainer.epochs=10\
    --overrides seed=1
