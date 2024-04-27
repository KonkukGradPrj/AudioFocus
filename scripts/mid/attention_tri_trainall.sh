#!/bin/bash
cd ../../

python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=midterm_v3 \
    --overrides exp_name=attention_tri_trainall\
    --overrides trainer.loss=tri\
    --overrides trainer.epochs=40\
    --overrides trainer.filter_every=true \
    --overrides seed=1
