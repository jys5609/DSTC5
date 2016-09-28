#!/bin/bash

dataset="dstc5_test"
dataroot="data"
ontology="scripts/config/ontology_dstc5.json"
trackfile=$1
scorefile="${trackfile%json}score.csv"

python scripts/check_main.py \
    --dataset $dataset --dataroot $dataroot --trackfile $trackfile \
    --ontology $ontology

python scripts/score_main.py \
    --dataset $dataset --dataroot $dataroot --trackfile $trackfile \
    --ontology $ontology --scorefile $scorefile

python scripts/report_main.py --scorefile $scorefile
