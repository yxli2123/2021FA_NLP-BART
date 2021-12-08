nohup python main.py \
--mode train \
--expt_dir /data1/yixiao/MNLI \
--expt_name xlnet-base \
--run_name bs_32 \
--model xlnet-base-cased \
--seq_len 256 \
--dataset multi_nli \
--seed 888 \
--lr 1e-5 \
--epochs 10 \
--batch_size 32 \
--gpu_id 3 &

