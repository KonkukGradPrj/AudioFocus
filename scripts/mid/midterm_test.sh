#!/bin/bash
cd ../../


# python run.py \
#     --config_name whisper_attention_titanet\
#     --overrides group_name=midterm_lastv \
#     --overrides exp_name=attention_titanet_one_20\
#     --overrides trainer.filter_every=true \
#     --overrides trainer.loss=tri\
#     --overrides trainer.epochs=20\
#     --overrides trainer.beta=1.0\
#     --overrides seed=1

python run.py \
    --config_name whisper_attention_titanet\
    --overrides group_name=midterm_lastv2_test_attenbias \
    --overrides exp_name=attention_titanet_1point0_20\
    --overrides trainer.filter_every=true \
    --overrides trainer.loss=tri\
    --overrides trainer.optimizer_type=sgd\
    --overrides trainer.epochs=5\
    --overrides trainer.beta=1\
    --overrides seed=1

# python run.py \
#     --config_name whisper_attention_titanet\
#     --overrides group_name=midterm_lastv2 \
#     --overrides exp_name=attention_titanet_0point3_20\
#     --overrides trainer.filter_every=true \
#     --overrides trainer.loss=tri\
#     --overrides trainer.epochs=5\
#     --overrides trainer.beta=0.3\
#     --overrides seed=1

# python run.py \
#     --config_name whisper_linear_titanet\
#     --overrides group_name=midterm_lastv2 \
#     --overrides exp_name=linear_titanet\
#     --overrides trainer.loss=tri\
#     --overrides trainer.epochs=10\
#     --overrides seed=1
