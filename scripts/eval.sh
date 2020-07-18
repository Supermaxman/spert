#!/usr/bin/env bash

python spert.py eval --config=configs/mm_umls_i2b2_2010_eval_v11.conf
python spert.py eval --config=configs/mm_umls_i2b2_2010_eval_v12.conf
python spert.py eval --config=configs/mm_umls_i2b2_2010_eval_v13.conf

python spert.py eval --config=configs/i2b2_2010_baseline_eval.conf
python spert.py eval --config=configs/i2b2_2010_baseline_base_eval.conf