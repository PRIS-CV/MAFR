# overall
CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --train_stage 1 --seed 0 --work_dirs experiments/output_mvcars --model_path checkpoints/resnet50.pth --temperature 1
CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --train_stage 2 --seed 0 --work_dirs experiments/output_mvcars --checkpoint_path experiments/output_mvcars/train-stage1/model_best.pth.tar --current_rewards True --same_gru True
CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --train_stage 3 --seed 0 --work_dirs experiments/output_mvcars --checkpoint_path experiments/output_mvcars/train-stage2/model_best.pth.tar --same_gru True