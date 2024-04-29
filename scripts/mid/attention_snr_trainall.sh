#!/bin/bash
cd ../../

python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=midterm_v3 \
    --overrides exp_name=attention_l1_trainall\
    --overrides trainer.loss=snr\
    --overrides trainer.epochs=40\
    --overrides trainer.filter_every=true \
    --overrides seed=1
