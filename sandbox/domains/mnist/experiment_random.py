from models import to_torch, FCNet, CNN1
import numpy as np
from data import gen_train_data, gen_test_data, WSampler

X_full, Y_full = gen_train_data(list(range(60000)))
X_half, Y_half = gen_train_data(list(range(30000)))
X_test, Y_test = gen_test_data()
X_test = X_test.reshape(-1, 28 * 28)

stop_criteria = (0.01, 1000, 120, 100000)
models = [
        lambda : FCNet(28*28, 10, stop_criteria).cuda(),
        lambda : CNN1((1,28,28), 10, stop_criteria).cuda(),
        ]

for X, Y in [(X_full, Y_full), (X_half, Y_half)]:
    for model in models:
        model = model()
        X = X.reshape(-1, 28*28)
        sampler = WSampler(X, Y, np.ones(shape=(60000,)))

        model.learn(sampler)

        score = model.evaluate((X_test, Y_test))
        print (score, "with ", len(Y))

