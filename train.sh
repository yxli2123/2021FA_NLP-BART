#!/usr/bin/env bash

python main.py \
--mode train \
--expt_dir /data1/yixiao/CNN \
--expt_name t5-large \
--run_name bs_4_1213 \
--model t5-large \
--seq_len 800 \
--dataset cnn_dailymail \
--seed 888 \
--lr 1e-5 \
--epochs 10 \
--batch_size 4 \
--gpu_id 3

python main.py \
--mode train \
--expt_dir /data1/yixiao/CNN \
--expt_name bart-large-cnn \
--run_name bs_2 \
--model facebook/bart-large-cnn \
--seq_len 800 \
--dataset cnn_dailymail \
--seed 888 \
--lr 1e-5 \
--epochs 10 \
--batch_size 2 \
--gpu_id 3 & 

