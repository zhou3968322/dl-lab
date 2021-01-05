#dl4
CUDA_VISIBLE_DEVICES=9,8 python train.py --config_file config/gan/medfe_mask_decoder.yaml --trainer_name medfe

CUDA_VISIBLE_DEVICES=8,7 python train.py --config_file config/gan/medfe_train.yaml --trainer_name medfe

# tensorboard
tensorboard --logdir tensorboard/medfe2 --port 8120 --bind_all
