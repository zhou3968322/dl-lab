CUDA_VISIBLE_DEVICES=7,8 python train.py --config_file config/gan/medfe_train.yaml --trainer_name medfe
CUDA_VISIBLE_DEVICES=7,8 python train.py --config_file config/gan/medfe_train2.yaml --trainer_name medfe

tensorboard --logdir tensorboard/medfe --port 8120 --bind_all
