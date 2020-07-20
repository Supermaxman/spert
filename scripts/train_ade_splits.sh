#!/usr/bin/env bash
# skips first split assuming that was trained already
for i in {0..9}; do
  python spert.py train --config=configs/ade/ade-s$i-mm-umls-train.conf
done
