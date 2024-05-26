#!/bin/bash
cd ../../



betas=(1 10 100)

for beta in "${betas[@]}"; do
    python run.py \
        --config_name whisper_attention_titanet\
        --overrides group_name=midterm_last_atten_betas_tune_hqd \
        --overrides exp_name=b_${beta}\
        --overrides trainer.filter_every=true \
        --overrides trainer.loss=tri\
        --overrides trainer.optimizer_type=sgd\
        --overrides trainer.epochs=3\
        --overrides trainer.beta=${beta}\
        --overrides seed=1
done    