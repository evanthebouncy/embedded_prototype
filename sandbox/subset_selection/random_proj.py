import numpy as np
import pickle
from sklearn import random_projection
T = random_projection.GaussianRandomProjection(n_components = 10)
X = np.random.rand(100, 1000)
X_new = T.fit_transform(X)
print (X_new.shape)
print (X.shape)

def project(X,dim = 32,loop = 10000):
    T = random_projection.GaussianRandomProjection(n_components=dim)
    
    X_new = []
    for i in range(0,X.shape[0],loop):
        X_new.append(T.fit_transform(X[i:i+loop]))
    X_new = np.vstack(X_new)
    
    return X_new

def flatten(X):
    s = X.shape
    s_new = (s[0],s[1]*s[2]*s[3])
    return np.reshape(X,s_new)

def get_X_Y(filename):
    with open(filename, 'rb') as f:
        X=pickle.load(f)
        Y=pickle.load(f)
    return X,Y
def save_X_Y(filename,X,Y):
    with open(filename,'wb') as f:
        pickle.dump((X,Y),f,protocol=4)
    return
if __name__ == '__main__':
    X,Y = get_X_Y('domains/pong/baselines/baselines/ppo2_memory_obs_actions')
    print(X.shape,Y.shape)
    X_flat=flatten(X)
    X_emb=project(X_flat)
    print(X_emb.shape)
    save_X_Y('domains/pong/baselines/baselines/ppo2_memory_dim32',X_emb,Y)

