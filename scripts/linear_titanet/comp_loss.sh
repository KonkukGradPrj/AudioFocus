#!/bin/bash
cd ../../
loss_fn=(all l2 l1)
# best l2, lr 1e-6
for loss in "${loss_fn[@]}"; do
    python run.py \
        --overrides group_name=tune_reslinear_titanet_0425 \
        --overrides exp_name=comp_loss_${loss} \
        --overrides trainer.loss=${loss}\
        --overrides seed="1"
done