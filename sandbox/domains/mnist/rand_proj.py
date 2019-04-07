import numpy as np
from sklearn import random_projection

if __name__ == '__main__':
    print ('random projecting mnist now')
    from data import gen_train_data, gen_test_data, WSampler
    X_tr, Y_tr = gen_train_data(list(range(60000)))
    X_tr = X_tr.reshape(-1, 28*28)

    emb_dim = 32

    import pickle

    T = random_projection.GaussianRandomProjection(n_components = emb_dim)
    X_tr_emb = T.fit_transform(X_tr)


    data_embed_path = 'mnist_rproj_{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )
