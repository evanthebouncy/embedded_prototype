from keras.datasets import mnist
import numpy as np

(X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
X_tr, X_t = X_tr / 255, X_t / 255

def gen_train_data(sub_idx):
    return X_tr[sub_idx], Y_tr[sub_idx]

def gen_test_data():
    return X_t, Y_t

import numpy as np

# the sampler for data, initialized with weights
class WSampler:

    # make the sampler
    def __init__(self, X, Y, W, data_aug = None):
        self.X, self.Y, self.W = X, Y, W
        self.data_aug = data_aug

    def get_sample(self, n):
        W = self.W
        if n > len(W):
            n = len(W)
        prob = np.array(W) / np.sum(W)
        r_idx = np.random.choice(range(len(W)), size=n, replace=True, p=prob)

        if self.data_aug is None:
            return self.X[r_idx], self.Y[r_idx]

        else:
            X_sub, Y_sub = self.X[r_idx], self.Y[r_idx]

            X_sub_aug = self.data_aug(X_sub)
            
            return X_sub_aug, Y_sub
