#!/usr/bin/env bash
# $1 name of experiment, eg i2b2-biobert
cat data/results/$1-*-eval-test.csv | awk 'NR % 2 == 0'