#!/usr/bin/env bash

#
for seed in {1..5}; do
  for split in {0..9}; do
    bash scripts/train_and_eval.sh \
      ade-$1-${split} \
      configs/ade/ade-$1 \
      ${seed} \
      data/datasets/ade/ade_split_${split}_train.json \
      data/datasets/ade/ade_split_${split}_test.json \
      data/datasets/ade/ade_split_${split}_test.json
  done
done
