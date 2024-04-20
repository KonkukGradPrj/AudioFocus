#!/bin/bash
cd ../../
loss_fn=(l2 l1 all)
lrs=(0.001 0.0001 0.00001 0.000001)

for loss in "${loss_fn[@]}"; do
    for lr in "${lrs[@]}"; do
        python run.py \
            --overrides group_name=attention_titanet_filter_every_0419 \
            --overrides exp_name=filter_every_train_from_front \
            --overrides trainer.loss=${loss}\
            --overrides trainer.filter_every=true \
            --overrides trainer.epochs=1 \
            --overrides trainer.optimizer.lr=${lr} \
            --overrides seed=1
    done
done
