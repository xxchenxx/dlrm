CUDA_VISIBLE_DEVICES=0 python dlrm_s_caffe2.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --use-gpu 



CUDA_VISIBLE_DEVICES=0 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=512 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed1.pth.tar --numpy-rand-seed 1 --use-gpu --mlperf-logging --tensor-board-filename seed1  --test-freq 512 > 0605_seed1_gpu0.out &


CUDA_VISIBLE_DEVICES=1 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=512 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed2.pth.tar --numpy-rand-seed 2 --use-gpu --mlperf-logging --tensor-board-filename seed2  --test-freq 512 > 0605_seed2_gpu1.out &

CUDA_VISIBLE_DEVICES=0 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=512 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed3.pth.tar --numpy-rand-seed 3 --use-gpu --mlperf-logging --tensor-board-filename seed3  --test-freq 512 > 0605_seed3_gpu0.out &



CUDA_VISIBLE_DEVICES=0 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=512 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed1_delta.pth.tar --numpy-rand-seed 1 --use-gpu --mlperf-logging --tensor-board-filename seed1_delta  --test-freq 512 > 0606_seed1_delta_gpu0.out &


CUDA_VISIBLE_DEVICES=0 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=512 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed1_delta.pth.tar --numpy-rand-seed 1 --use-gpu --mlperf-logging --tensor-board-filename seed1_delta  --test-freq 512 > 0606_seed1_delta_gpu0.out &

CUDA_VISIBLE_DEVICES=1 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=512 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed2_delta.pth.tar --numpy-rand-seed 2 --use-gpu --mlperf-logging --tensor-board-filename seed2_delta  --test-freq 512 > 0606_seed2_delta_gpu1.out &



CUDA_VISIBLE_DEVICES=0 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=512 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed1_delta.pth.tar --numpy-rand-seed 1 --use-gpu --mlperf-logging --tensor-board-filename seed1_delta  --test-freq 512 > 0608_seed1_delta_gpu0.out &

CUDA_VISIBLE_DEVICES=1 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=512 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed2_delta.pth.tar --numpy-rand-seed 2 --use-gpu --mlperf-logging --tensor-board-filename seed2_delta  --test-freq 512 > 0608_seed2_delta_gpu1.out 

CUDA_VISIBLE_DEVICES=6,7 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model model_seed1_delta.pth.tar --numpy-rand-seed 1 --use-gpu --tensor-board-filename seed3_delta  --test-freq 1024 > 0609_seed1_delta_gpu67.out &



CUDA_VISIBLE_DEVICES=4,5 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed1 --numpy-rand-seed 1 --use-gpu --tensor-board-filename seed1  --test-freq 1024 > 0610_seed1_gpu45.out &

CUDA_VISIBLE_DEVICES=6,7 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed2 --numpy-rand-seed 2 --use-gpu --tensor-board-filename seed2  --test-freq 1024 > 0610_seed1_gpu67.out &

CUDA_VISIBLE_DEVICES=0,1 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed2 --numpy-rand-seed 3 --use-gpu --tensor-board-filename seed3  --test-freq 1024 > 0610_seed3_gpu01.out &

CUDA_VISIBLE_DEVICES=2,3 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed2 --numpy-rand-seed 3 --use-gpu --tensor-board-filename seed4  --test-freq 1024 > 0610_seed4_gpu23.out &

CUDA_VISIBLE_DEVICES=0,1 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.5 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed1_0.5 --numpy-rand-seed 1 --use-gpu --tensor-board-filename seed1_0.5  --test-freq 1024 > 0610_seed1_lr0.5_gpu01.out &

CUDA_VISIBLE_DEVICES=2,3 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed1_1 --numpy-rand-seed 1 --use-gpu --tensor-board-filename seed1_1  --test-freq 1024 > 0610_seed1_lr1_gpu23.out &







CUDA_VISIBLE_DEVICES=4,5 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed1 --numpy-rand-seed 1 --use-gpu --tensor-board-filename seed1  --test-freq 1024 > 0611_seed1_gpu45.out &

CUDA_VISIBLE_DEVICES=6,7 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed2 --numpy-rand-seed 2 --use-gpu --tensor-board-filename seed2  --test-freq 1024 > 0611_seed1_gpu67.out &

CUDA_VISIBLE_DEVICES=0,1 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed2 --numpy-rand-seed 3 --use-gpu --tensor-board-filename seed3  --test-freq 1024 > 0611_seed3_gpu01.out &

CUDA_VISIBLE_DEVICES=2,3 nohup python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_seed2 --numpy-rand-seed 3 --use-gpu --tensor-board-filename seed4  --test-freq 1024 > 0611_seed4_gpu23.out &