#!/bin/bash
cd ../../

python run.py \
    --overrides group_name=test \
    --overrides exp_name=test \
    --overrides trainer.loss=l2\
    --overrides trainer.filter_every=true \
    --overrides seed=1

### 이거 test wer 0.7나옴
cd ../../
loss_fn=(all)
for loss in "${loss_fn[@]}"; do
    python run.py \
        --overrides group_name=tune_reslinear_titanet_0425 \
        --overrides exp_name=comp_loss_${loss} \
        --overrides trainer.loss=${loss}\
        --overrides seed="1"
done