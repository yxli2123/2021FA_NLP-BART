python main.py \
--mode train \
--expt_dir /data1/yixiao/ \
--expt_name CNN \
--run_name bs_1_test \
--model roberta-large \
--seq_len 256 \
--dataset multi_nli \
--seed 888 \
--lr 1e-5 \
--epochs 10 \
--batch_size 1 \
--gpu_id 2
