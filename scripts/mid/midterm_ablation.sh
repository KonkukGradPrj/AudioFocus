#!/bin/bash
cd ../../


# python run.py \
#     --config_name whisper_attention_titanet\
#     --overrides group_name=midterm_ablation_V2 \
#     --overrides exp_name=vanilla\
#     --overrides trainer.filter_every=true \
#     --overrides trainer.optimizer_type=sgd\
#     --overrides trainer.loss=tri\
#     --overrides trainer.epochs=10\
#     --overrides trainer.beta=1\
#     --overrides seed=1

# python run.py \
#     --config_name whisper_attention_titanet\
#     --overrides group_name=midterm_ablation_V2 \
#     --overrides exp_name=no_filter_every\
#     --overrides trainer.optimizer_type=sgd\
#     --overrides trainer.filter_every=false \
#     --overrides trainer.loss=tri\
#     --overrides trainer.epochs=1\
#     --overrides trainer.beta=1\
#     --overrides seed=1


python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=midterm_ablation_V2 \
    --overrides exp_name=no_beta\
    --overrides trainer.optimizer_type=sgd\
    --overrides trainer.filter_every=true \
    --overrides trainer.loss=tri\
    --overrides trainer.epochs=1\
    --overrides trainer.beta=0\
    --overrides seed=1

python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=midterm_ablation_V2 \
    --overrides trainer.optimizer_type=sgd\
    --overrides exp_name=no_beta\
    --overrides trainer.filter_every=true \
    --overrides trainer.loss=tri\
    --overrides trainer.epochs=1\
    --overrides trainer.beta=0\
    --overrides seed=1