CUDA_VISIBLE_DEVICES=0,1 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.5 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed1_sgd_lr0.5 --numpy-rand-seed 1 --use-gpu --tensor-board-filename seed1_sgd_lr0.5  --test-freq 1024 > 0611_seed1_sgd_lr0.5_gpu01.out &