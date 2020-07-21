#!/usr/bin/env bash
# $1 name of experiment, eg biobert
cat data/results/i2b2-$1-*-eval-test.csv | awk 'NR % 2 == 0'
