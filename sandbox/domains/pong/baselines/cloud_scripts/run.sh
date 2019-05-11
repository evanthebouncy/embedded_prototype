#!/bin/bash
source ~/.bashrc
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e5 --save_path=../pong_data/ppo2_pong_model_1e5 --mode=train_expert
# sudo poweroff

