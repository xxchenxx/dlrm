#0611_add_label_noise_lr0.1_ln0.01_sgd_seed1_gpu1.sh
exp_name=add_label_noise
lr=0.1
label_noise=0.01
optimizer=sgd
seed=1
gpu=1


CUDA_VISIBLE_DEVICES=${gpu} nohup python dlrm_s_pytorch_add_label_noise.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=${lr} --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model-dir model_${exp_name}_lr${lr}_wn${label_noise}_${optimizer}_seed${seed} --numpy-rand-seed ${seed} --use-gpu --tensor-board-filename ${exp_name}_lr${lr}_wn${label_noise}_${optimizer}_seed${seed}  --test-freq 1024 --optimizer ${optimizer} --label-noise ${label_noise} > 0611_${exp_name}_lr${lr}_wn${label_noise}_${optimizer}_seed${seed}_gpu${gpu}.out &


