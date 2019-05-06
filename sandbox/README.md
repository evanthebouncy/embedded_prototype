# Running the PPO experiments
    
go to first /baselines
    
# train the expert model

    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=ppo2_pong 'raw'
    
# use the expert model to generate a set of (obs, returns, masks, actions, values, neglogpacs)

    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e6 --load_path=ppo2_pong 'memory'
    
# use the whole memory to train a model
    
    python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=8e6 --save_path=useless 'pretrain_whole'
    
    # go to baselines
    # use baselines/inspector.py to transform the memory to two X and Y
    # go to pong
    # use cnn_vae.py to embed the memory into data_embed_path
    # go to sandbox
    # run python -m subset_selection.make_all_subset_batches tiers ./domains/pong/pong_emb_32.p ./domains/pong/pong_tiers_32.p
    # go to baselines.
    # run the baselines.run files again. 
