import numpy as np
from sklearn import random_projection
T = random_projection.GaussianRandomProjection(n_components = 10)
X = np.random.rand(100, 1000)
X_new = T.fit_transform(X)
print (X_new.shape)
print (X.shape)

