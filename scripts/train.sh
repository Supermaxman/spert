#!/usr/bin/env bash

python spert.py train --config=configs/mm_umls_train_v11.conf
python spert.py train --config=configs/mm_umls_train_v12.conf
python spert.py train --config=configs/mm_umls_train_v13.conf

mv data/save/mm-umls-train-st21pv-v11/*/final_model data/save/mm-umls-train-st21pv-v11/final_model/
mv data/save/mm-umls-train-st21pv-v12/*/final_model data/save/mm-umls-train-st21pv-v12/final_model/
mv data/save/mm-umls-train-st21pv-v13/*/final_model data/save/mm-umls-train-st21pv-v13/final_model/

python spert.py train --config=configs/mm_umls_i2b2_2010_train_v11.conf
python spert.py train --config=configs/mm_umls_i2b2_2010_train_v12.conf
python spert.py train --config=configs/mm_umls_i2b2_2010_train_v13.conf

python spert.py train --config=configs/i2b2_2010_baseline_train.conf
python spert.py train --config=configs/i2b2_2010_baseline_base_train.conf

mv data/save/mm-umls-i2b2-2010-train-v11/*/final_model data/save/mm-umls-i2b2-2010-train-v11/final_model/
mv data/save/mm-umls-i2b2-2010-train-v12/*/final_model data/save/mm-umls-i2b2-2010-train-v12/final_model/
mv data/save/mm-umls-i2b2-2010-train-v13/*/final_model data/save/mm-umls-i2b2-2010-train-v13/final_model/


mv data/save/i2b2-2010-baseline-train-v1/*/final_model data/save/i2b2-2010-baseline-train-v1/final_model/
mv data/save/i2b2-2010-baseline-base-train-v1/*/final_model data/save/i2b2-2010-baseline-base-train-v1/final_model/

python spert.py eval --config=configs/