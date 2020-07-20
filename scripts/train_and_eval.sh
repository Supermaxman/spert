#!/usr/bin/env bash

# $1 is the model name
# $2 is the config path
# $3 is the seed number
# $4 is the train path
# $5 is the val path
# $6 is the test path

python spert.py train \
  --label=$1-$3-train \
  --config=$2-train.conf \
  --seed=$3 \
  --train_path=$4 \
  --valid_path=$5

mv data/save/$1-$3-train/*/final_model data/save/$1-$3-train/final_model/

python spert.py eval \
 --label=$1-$3-eval \
 --config=$2-eval.conf \
 --seed=$3 \
 --model_path=data/save/$1-$3-train/final_model/ \
 --tokenizer_path=data/save/$1-$3-train/final_model/ \
 --dataset_path=$6

mv data/save/$1-$3-eval/*/final_model data/save/$1-$3-eval/final_model/
cp data/save/$1-$3-eval/final_model/eval_test.csv data/results/$1-$3-eval-test.csv