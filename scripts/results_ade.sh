#!/usr/bin/env bash
# $1 name of experiment, eg biobert
for split in {0..9}; do
    cat data/results/ade-$1-${split}-*-eval-test.csv | awk 'NR % 2 == 0'
done