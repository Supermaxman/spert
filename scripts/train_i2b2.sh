#!/usr/bin/env bash

#
for seed in {0..4}; do
  bash scripts/train_and_eval.sh \
    i2b2-$1 \
    configs/i2b2/i2b2-$1 \
    ${seed} \
    data/i2b2/train.json \
    data/i2b2/dev.json \
    data/i2b2/test.json
done
