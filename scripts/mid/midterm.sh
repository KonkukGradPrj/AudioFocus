#!/bin/bash
cd ../../
loss_fn=(all l2 l1)


for loss in "${loss_fn[@]}"; do
    python run.py \
        --config_name whisper_attention_titanet\
        --overrides group_name=midterm_v2 \
        --overrides exp_name=attention_${loss}_init0\
        --overrides trainer.filter_every=true \
        --overrides trainer.loss=${loss}\
        --overrides trainer.epochs=40\
        --overrides seed=1
done

for loss in "${loss_fn[@]}"; do
    python run.py \
        --config_name whisper_linear_titanet\
        --overrides group_name=midterm_v2 \
        --overrides exp_name=linear_${loss}_init0\
        --overrides trainer.loss=${loss}\
        --overrides trainer.epochs=10\
        --overrides seed=1
done
