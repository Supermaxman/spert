#!/usr/bin/env bash

# i2b2 2010/VA public corpus
# bash scripts/train_i2b2.sh bert-base
# bash scripts/train_i2b2.sh biobert
# bash scripts/train_i2b2.sh ncbi-bert

# Adverse Drug Effect corpus
bash scripts/train_ade.sh bert-base
bash scripts/train_ade.sh biobert
bash scripts/train_ade.sh ncbi-bert
