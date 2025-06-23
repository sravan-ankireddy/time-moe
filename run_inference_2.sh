CUDA_VISIBLE_DEVICES=1 python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 256

CUDA_VISIBLE_DEVICES=1 python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 512

CUDA_VISIBLE_DEVICES=1 python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 1024

CUDA_VISIBLE_DEVICES=1 python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 2048 -b 16

CUDA_VISIBLE_DEVICES=1 python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 4096 -b 8