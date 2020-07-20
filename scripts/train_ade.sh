#!/usr/bin/env bash

#
num_seeds=5
max_split=9
for seed in {1..${num_seeds}}; do
  for split in {0..${max_split}}; do
    bash scripts/train_and_eval.sh \
      ade-$1-${split} \
      configs/ade/ade-$1 \
      ${seed} \
      data/datasets/ade/ade_split_${split}_train.json \
      data/datasets/ade/ade_split_${split}_test.json \
      data/datasets/ade/ade_split_${split}_test.json
  done
done
