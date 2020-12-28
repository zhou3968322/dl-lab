# dl4
CUDA_VISIBLE_DEVICES=4,2 python train.py --config_file config/gan/medfe_train2.yaml --trainer_name medfe

# dl2
CUDA_VISIBLE_DEVICES=2,1 python train.py --config_file config/gan/medfe_train_together.yaml --trainer_name medfe # 无效的训练

CUDA_VISIBLE_DEVICES=9,8 python train.py --config_file config/gan/medfe_with_mask_decoder.yaml --trainer_name medfe

#dl4
CUDA_VISIBLE_DEVICES=8,7 python train.py --config_file config/gan/medfe_fake_mask_decoder.yaml --trainer_name medfe

# dl2

CUDA_VISIBLE_DEVICES=8,7 python train.py --config_file config/gan/medfe_with_inner_loss_fake_mask_decoder.yaml --trainer_name medfe

# tensorboard
tensorboard --logdir tensorboard/medfe2 --port 8120 --bind_all
