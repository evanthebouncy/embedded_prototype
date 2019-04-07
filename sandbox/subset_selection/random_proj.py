import numpy as np
import pickle
from sklearn import random_projection
T = random_projection.GaussianRandomProjection(n_components = 10)
X = np.random.rand(100, 1000)
X_new = T.fit_transform(X)
print (X_new.shape)
print (X.shape)

def project(X,dim = 32):
    T = random_projection.GaussianRandomProjection(n_components=dim)

    X_new = T.fit_transform(X)
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

if __name__ == '__main__':
    X,Y = get_X_Y('domains/pong/baselines/baselines/ppo2_memory')
    print(X.shape,Y.shape)
    X_flat=flatten(X)
    X_emb=project(X_flat)
    print(X_emb.shape)
