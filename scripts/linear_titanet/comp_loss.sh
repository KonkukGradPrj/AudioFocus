#!/bin/bash
cd ../../
loss_fn=(all distill l2 l1)
### gradient 가 존나 큼. => lr이 개짝아야함 대충 scale 이 5e5 => 1e4 정도... l2기준이고 distill은 훨씬 클듯..
for seed in {1..3}; do
    for loss in "${loss_fn[@]}"; do
        python run.py \
            --overrides group_name=tune_reslinear_titanet_0404 \
            --overrides exp_name=comp_loss_${loss} \
            --overrides trainer.loss=${loss}\
            --overrides seed="$seed"
    done
done