CUDA_VISIBLE_DEVICES=8,9 python train.py --config_file config/gan/medfe_train1.yaml --trainer_name medfe
CUDA_VISIBLE_DEVICES=8,9 python train.py --config_file config/gan/medfe_train.yaml --trainer_name medfe

CUDA_VISIBLE_DEVICES=6,7 python train.py --config_file config/gan/medfe_no_gaussian_train.yaml --trainer_name medfe
CUDA_VISIBLE_DEVICES=8,9 python train.py --config_file config/gan/medfe_train_together.yaml --trainer_name medfe
CUDA_VISIBLE_DEVICES=8,9 python train.py --config_file config/gan/medfe_train_lr.yaml --trainer_name medfe

CUDA_VISIBLE_DEVICES=5,7 python train.py --config_file config/gan/medfe_with_inner_loss.yaml --trainer_name medfe
tensorboard --logdir tensorboard/medfe2 --port 8120 --bind_all
