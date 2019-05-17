# Running the PPO experiments
    
go to first /baselines
    
# train the expert model

to accomplish this we set --mode=train\_expert, and specify a save\_path

note: if save\_path is specified, the model will be saved. otherwise it wont

    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=../pong_data/ppo2_pong_model --mode=train_expert
    
    # for CartPole-v0
    python -m baselines.run --alg=ppo2 --env=CartPole-v0 --num_timesteps=1e6 --save_path=../cart_data/ppo2_cart_model_1e6 --mode=train_expert
    
# use the expert model to generate a set of supervision memory

to accomplish this set --mode=collect\_supervision
and we specify a memory path as well

    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e5 --load_path=../pong_data/ppo2_pong_model --mode=collect_supervision --memory_path=../pong_data/ppo2_memory
    
    # for CartPole-v0
    python -m baselines.run --alg=ppo2 --env=CartPole-v0 --num_timesteps=1e5 --load_path=../cart_data/ppo2_cart_model_1e6 --mode=collect_supervision --memory_path=../cart_data/ppo2_memory
    

# go to baselines. Save XYs in baselines/baselines/ppo2_memory_obs_actions

    python baselines/inspector.py ../ppo2_memory baselines/ppo2_memory_obs_actions
    
    python inspector.py pong_data/ppo2_memory_1e5 pong_data/ppo2_memory_1e5_xy
    
to transform the memory to two X and Y
    
    
# go to pong. Save embeddings at pong/pong_emb_32.p

    python cnn_vae.py baselines/baselines/ppo2_memory_obs_actions pong_emb_32.p
   
    # xy path, xy_32 path, learn_iter, batch_size
    python cnn_vae.py pong_data/ppo2_memory_1e5_xy pong_data/ppo_1e5_32.p 100000 100
    
to embed the memory into data_embed_path. The first is the memory file generated above, and the second is the embedded file

# go to sandbox. Save index at pong/pong_tiers_32.p.
    
    python -m subset_selection.make_all_subset_batches tiers ./domains/pong/pong_emb_32.p ./domains/pong/pong_tiers_32.p

    # for CartPole-v0
    python -m subset_selection.make_all_subset_batches tiers ./domains/pong/cart_data/ppo2_memory_obs_actionstuple ./domains/pong/cart_data/pong_tiers_32.p
# go to pong. Save the three files in pong/pong_idx_whole.p,pong/pong_idx_random.p,pong/pong_idx_subset.p
    
    python generate_idx_pickle.py tot_size all sub_size pong_idx pong_tiers_32.p
    
    # for CartPole-v0
    python generate_idx_pickle.py 98304 all 10000 cart_data/pong_idx cart_data/pong_tiers_32.p
    
    Here, you need to figure out the tot_size and the sub_size yourself
    
    
# use the memory to train a model. Set the mode to "pretrain" in ppo2.py at baselines/baselines/ppo2/ppo2.py
Rename the pickle_path, and the loss_path depending on your need.

    # this
    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1000 --mode=evaluate_pretrain --index_path=../pong_data/idx_all.p --memory_path=../pong_data/ppo2_memory --loss_path=../pong_data/result_loss_idx_all

    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=useless --mode=pretrain --pickle_path=../pong_idx_whole.p --memory_path=../ppo2_memory --loss_path=ppo2_losses_with_whole_subset
    
    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=useless --mode=pretrain --pickle_path=../pong_idx_random.p --memory_path=../ppo2_memory --loss_path=ppo2_losses_with_random_subset
    
    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=useless --mode=pretrain --pickle_path=../pong_idx_subset.p --memory_path=../ppo2_memory --loss_path=ppo2_losses_with_selected_subset
    
    # for CartPole-v0
    
    python -m baselines.run --alg=ppo2 --env=CartPole-v0 --num_timesteps=1e5 --mode=evaluate_pretrain --index_path=../cart_data/pong_idx_whole.p --memory_path=../cart_data/ppo2_memory --loss_path=../cart_data/result_loss_idx_all --log_interval=1
    
    
    
    
    
    
