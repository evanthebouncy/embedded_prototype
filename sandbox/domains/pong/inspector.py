import pickle
import numpy as np
import argparse





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("load_path")
    parser.add_argument("save_path")
    args = parser.parse_args()

    path = args.load_path#'../ppo2_memory'

    with open(path, 'rb') as f:
        data = pickle.load(f)
    obs_ = []
    actions_ = []
    for obs, returns, masks, actions, values, neglogpacs in data:
        obs_.append(obs)
        actions_.append(actions)
    obs_ = np.vstack(obs_)
    actions_ = np.hstack(actions_)
    print(obs_.shape, actions_.shape)

    save_path = args.save_path #'baselines/ppo2_memory_obs_actions'

    with open(save_path, 'wb') as f:
        pickle.dump((obs_, actions_), f, protocol=4)
