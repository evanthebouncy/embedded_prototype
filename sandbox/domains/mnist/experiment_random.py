from models import to_torch, FCNet 
import numpy as np
from data import gen_train_data, gen_test_data, WSampler

stop_criteria = (0.01, 1000, 120, 100000)
fcnet = FCNet(28*28, 10, stop_criteria).cuda()
X, Y = gen_train_data(list(range(60000)))
X = X.reshape(-1, 28*28)
sampler = WSampler(X, Y, np.ones(shape=(60000,)))

fcnet.learn(sampler)

X_test, Y_test = gen_test_data()
X_test = X_test.reshape(-1, 28 * 28)
score = fcnet.evaluate((X_test, Y_test))
print (score)

