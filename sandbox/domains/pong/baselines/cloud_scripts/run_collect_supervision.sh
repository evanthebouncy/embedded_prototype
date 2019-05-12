python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e5 --load_path=../pong_data/ppo2_pong_model_8e6 --mode=collect_supervision --memory_path=../pong_data/ppo2_memory_1e5
sudo poweroff

