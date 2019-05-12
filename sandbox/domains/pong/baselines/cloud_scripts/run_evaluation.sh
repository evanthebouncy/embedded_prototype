python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --mode=evaluate_pretrain --index_path=../pong_data/idx_none.p --memory_path=../pong_data/ppo2_memory_1e5 --loss_path=../pong_data/result_idx_none

