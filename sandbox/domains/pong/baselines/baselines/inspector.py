import pickle

path = '='

with open(path,'rb') as f:
    data = pickle.load(f)
ans = []
for obs_t, action, reward, obs_tp1, done in data:
    ans.append((obs_t.__array__(),action,reward,obs_tp1.__array__(),done))
    print(obs_t.__array__().shape, action, reward, obs_tp1.__array__().shape, done)

with open('inspected_memory','wb') as f:
    pickle.dump(ans,f)
