# Running the PPO experiments
    
go to first /baselines
    
# train the expert model. Save the model at baselines/ppo2_pong

    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=ppo2_pong --mode=raw
    
# use the expert model to generate a set of (obs, returns, masks, actions, values, neglogpacs). Save memory at pong/ppo2_memory

    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e6 --load_path=ppo2_pong --mode=memory --memory_path=../ppo2_memory
   

# go to baselines. Save XYs in baselines/baselines/ppo2_memory_obs_actions

    python baselines/inspector.py ../ppo2_memory baselines/ppo2_memory_obs_actions
    
    to transform the memory to two X and Y
    
    
# go to pong. Save embeddings at pong/pong_emb_32.p

    python cnn_vae.py baselines/baselines/ppo2_memory_obs_actions pong_emb_32.p
    
    to embed the memory into data_embed_path. The first is the memory file generated above, and the second is the embedded file

# go to sandbox. Save index at pong/pong_tiers_32.p.
    
    python -m subset_selection.make_all_subset_batches tiers ./domains/pong/pong_emb_32.p ./domains/pong/pong_tiers_32.p


# go to pong. Save the three files in pong/pong_idx_whole.p,pong/pong_idx_random.p,pong/pong_idx_subset.p
    
    python generate_idx_pickle.py tot_size all sub_size pong_idx pong_tiers_32.p
    
    Here, you need to figure out the tot_size and the sub_size yourself
    
    
# use the memory to train a model. Set the mode to "pretrain" in ppo2.py at baselines/baselines/ppo2/ppo2.py
Rename the pickle_path, and the loss_path depending on your need.
    
    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=useless --mode=pretrain --pickle_path=../pong_idx_whole.p --memory_path=../ppo2_memory --loss_path=ppo2_losses_with_whole_subset
    
    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=useless --mode=pretrain --pickle_path=../pong_idx_random.p --memory_path=../ppo2_memory --loss_path=ppo2_losses_with_random_subset
    
    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=useless --mode=pretrain --pickle_path=../pong_idx_subset.p --memory_path=../ppo2_memory --loss_path=ppo2_losses_with_selected_subset
    
    
    
    
    
    
