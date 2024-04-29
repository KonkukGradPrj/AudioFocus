#!/bin/bash
cd ../../


python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=midterm_lastv \
    --overrides exp_name=attention_titanet_1.0_20\
    --overrides trainer.filter_every=true \
    --overrides trainer.loss=tri\
    --overrides trainer.epochs=20\
    --overrides trainer.beta=1.0\
    --overrides seed=1

python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=midterm_lastv \
    --overrides exp_name=attention_titanet_1.0_20\
    --overrides trainer.filter_every=true \
    --overrides trainer.loss=tri\
    --overrides trainer.epochs=20\
    --overrides trainer.beta=2.0\
    --overrides seed=1

python run.py \
    --config_name whisper_linear_titanet\
    --overrides group_name=midterm_lastv \
    --overrides exp_name=linear_titanet\
    --overrides trainer.loss=tri\
    --overrides trainer.epochs=10\
    --overrides seed=1
