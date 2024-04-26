#!/bin/bash
cd ../

python run.py \
    --config_name whisper_linear_titanet\
    --overrides group_name=test \
    --overrides exp_name=test \
    --overrides trainer.loss=l2\
    --overrides seed=1
    
python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=test \
    --overrides exp_name=test \
    --overrides trainer.loss=l2\
    --overrides trainer.filter_every=true \
    --overrides seed=1
