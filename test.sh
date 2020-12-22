#just for test
CUDA_VISIBLE_DEVICES=9,7 python train.py --config_file config/gan/medfe_train.yaml --trainer_name medfe
CUDA_VISIBLE_DEVICES=8,6 python train.py --config_file config/gan/medfe_train1.yaml --trainer_name medfe
CUDA_VISIBLE_DEVICES=4,2 python train.py --config_file config/gan/medfe_train2.yaml --trainer_name medfe

# dl4
CUDA_VISIBLE_DEVICES=9,7 python train.py --config_file config/gan/medfe_train.yaml --trainer_name medfe

# CUDA_VISIBLE_DEVICES=8,6 python train.py --config_file config/gan/medfe_test1.yaml --trainer_name medfe
CUDA_VISIBLE_DEVICES=8,6 python train.py --config_file config/gan/medfe_train_together.yaml --trainer_name medfe

CUDA_VISIBLE_DEVICES=4,5 python train.py --config_file config/gan/medfe_with_mask_decoder.yaml --trainer_name medfe


# dl2
CUDA_VISIBLE_DEVICES=9,8 python train.py --config_file config/gan/medfe_with_inner_loss.yaml --trainer_name medfe

CUDA_VISIBLE_DEVICES=7,5 python train.py --config_file config/gan/medfe_with_inner_loss_mask_decoder.yaml --trainer_name medfe

# tensorboard
tensorboard --logdir tensorboard/medfe2 --port 8120 --bind_all
