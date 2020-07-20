#!/usr/bin/env bash

#
num_seeds=5
for seed in {1..${num_seeds}}; do
  bash scripts/train_and_eval.sh \
    i2b2-$1 \
    configs/i2b2/i2b2-$1 \
    ${seed} \
    /users/max/data/corpora/i2b2/2010/json/train.json \
    /users/max/data/corpora/i2b2/2010/json/dev.json \
    /users/max/data/corpora/i2b2/2010/json/test.json
done
