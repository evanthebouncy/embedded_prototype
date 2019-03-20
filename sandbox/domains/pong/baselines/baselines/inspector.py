import pickle
import numpy as np

path = '../ppo2_memory'

with open(path,'rb') as f:
    data = pickle.load(f)
obs_ = []
actions_ = []
for obs,returns,masks,actions,values,neglogpacs in data:
    obs_.append(obs)
    actions_.append(actions)
obs_ = np.vstack(obs_)
actions_ = np.hstack(actions_)
print(obs_.shape,actions_.shape)



with open('baselines/ppo2_memory_obs_actions','wb') as f:
    pickle.dump(obs_,f,protocol=4)
    pickle.dump(actions_,f,protocol=4)
