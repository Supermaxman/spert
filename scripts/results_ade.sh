#!/usr/bin/env bash
# $1 name of experiment, eg biobert
for seed in {0..4}; do
    cat data/results/ade-$1-*-${seed}-eval-test.csv | awk 'NR % 2 == 0'
done