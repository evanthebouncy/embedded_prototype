python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e6 --mode=evaluate_pretrain --index_path=../pong_data/idx_all.p --memory_path=../pong_data/ppo2_memory --loss_path=../pong_data/result_loss_idx_all
sudo poweroff

