#!/bin/bash
cd ../../

betas=(0.0 -0.3 -0.5 -0.8)

for beta in "${betas[@]}"; do
    python run.py \
        --config_name whisper_attention_titanet\
        --overrides group_name=tune_beta \
        --overrides exp_name=tune_beta_${beta}\
        --overrides trainer.loss=tri\
        --overrides trainer.epochs=1\
        --overrides trainer.beta=${beta}\
        --overrides trainer.filter_every=true \
        --overrides seed=1 
done
