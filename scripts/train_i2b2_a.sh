#!/usr/bin/env bash

#
seed=0
#for seed in {0..4}; do
  bash scripts/train_and_eval.sh \
    i2b2-$1 \
    configs/i2b2/i2b2-$1 \
    ${seed} \
    /users/max/data/corpora/i2b2/2010/json2/train.json \
    /users/max/data/corpora/i2b2/2010/json2/dev.json \
    /users/max/data/corpora/i2b2/2010/json2/test.json
#done
