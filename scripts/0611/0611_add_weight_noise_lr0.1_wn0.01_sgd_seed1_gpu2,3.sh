#0611_add_weight_noise_lr0.1_wn0.01_sgd_seed1_gpu2,3.sh
exp_name=add_weight_noise
lr=0.1
weight_noise=0.01
optimizer=sgd
seed=1
gpu=2,3


CUDA_VISIBLE_DEVICES=${gpu} nohup python python dlrm_s_pytorch_add_weight_noise.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=${lr} --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_${exp_name}_lr${lr}_wn${weight_noise}_${optimizer}_seed${seed} --numpy-rand-seed ${seed} --use-gpu --tensor-board-filename ${exp_name}_lr${lr}_wn${weight_noise}_${optimizer}_seed${seed}  --test-freq 1024 --optimizer ${optimizer} --weight-noise ${weight_noise} > 0611_${exp_name}_lr${lr}_wn${weight_noise}_${optimizer}_seed${seed}_gpu${gpu}.out &


