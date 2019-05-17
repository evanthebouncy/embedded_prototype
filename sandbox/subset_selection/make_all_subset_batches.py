import numpy as np
import random
import time
import math
from copy import deepcopy
from sklearn.cluster import KMeans
from .knn import score_subset, update_weight
from sklearn.neighbors import kneighbors_graph
from collections import Counter

from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial import cKDTree
from .rec_annealing import recover_index
from .condense import condense_once
import argparse

# perform condensor algorithm and save to save_path

def get_newXY(X,Y,path):
    with open(path,'rb') as f:
        idx = pickle.load(f)
    rm = []
    for i in range(len(idx[0])):
        print(len(idx[0][i]))
        rm.extend(idx[0][i])
    X_sub, Y_sub = np.delete(X, rm, 0), np.delete(Y, rm)
    return X_sub,Y_sub,idx[0],idx[1]


def condensor(args):
    import time
    import pickle
    from tqdm import tqdm

    X_tr_emb, Y_tr = args.X_tr_emb, args.Y_tr
    save_path = args.save_path
    final_size = args.final_size
    throw_frac = args.throw_frac
    require_loss = args.require_loss

    print (" =========================================================== ")
    print ("running condensor with following args" )
    print ("X shape ", X_tr_emb.shape)
    print ("Y shape ", Y_tr.shape)
    print ("save_path ", save_path)
    print ("final_size ", final_size)
    print ("throw_frac ", throw_frac)
    print ("require loss ", require_loss)

    N = X_tr_emb.shape[0]

    estimate_iter = math.log(final_size / N, (1.0 - throw_frac))
    print ("estimated iteration ", estimate_iter)

    index_rankings = []
    losses = []
    X, Y = X_tr_emb, Y_tr
    # X,Y,index_rankings,losses = get_newXY(X,Y)
    print(X.shape,Y.shape,X_tr_emb.shape)
    for i in tqdm(range(int(estimate_iter + 100))):
        X, Y, rm_idx, loss = condense_once(X, Y, X_tr_emb, Y_tr, throw_frac, require_loss)
        print("iteration ", i, " cur size ", len(Y), 'loss ',loss)
        losses = [loss] + losses
        index_rankings = [rm_idx] + index_rankings
        # add in the last bit of indexes
        if X.shape[0] < final_size:
            rm_idx = recover_index(X, X_tr_emb)
            index_rankings = [rm_idx] + index_rankings
            losses = [0] + losses 

        pickle.dump((index_rankings, losses), open(save_path, "wb"))
        print ("saved rankings/losses at ", save_path)

        if X.shape[0] < final_size:
            break

    print ("all done ! ")



def kmeans(X_tr_emb,Y_tr,name,size = 100):
    import time
    import pickle
    from tqdm import tqdm
    from subset_selection.rec_annealing import k_means_idx

    #print("loading them pickle . . . ")
    #data_embed_path = 'data_embed/mnist_dim32.p'
    #X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
    # X_tr_emb, Y_tr = X_tr_emb[:1000],Y_tr[:1000]
    print("loaded ")
    ans = []
    #size = 100
    fract = 1.2
    while (size < 15000):
        print(size)
        ans.append(k_means_idx(X_tr_emb, int(size)))
        size = size * fract
        data_tier_path = 'data_sub/'+name+'_kmeans.p'
        pickle.dump(np.array(ans), open(data_tier_path, "wb"))
        print("saved . . .", data_tier_path)
        # print(ans)

def flat_anneal(X_tr_emb,Y_tr,idx_condensor,name,size = 100):
    import pickle
    from subset_selection.rec_annealing import anneal_optimize
    #data_path = 'data_sub/mnist_tiers.p'
    #idx_condensor = index
    #print(idx_condensor.shape)
    X, Y = X_tr_emb, Y_tr

    #size = 100
    frac = 1.1
    cond_anneal = []

    while size < idx_condensor.shape[0]:
        index = idx_condensor[:size]
        tl = 60
        #if size == 100:
        #    tl = 300
        cond_anneal.append(anneal_optimize(index, X, Y, tl))
        pickle.dump(cond_anneal, open('data_sub/'+name+'_tiers_anneal.p', 'wb'))
        size = int(size * frac)
        print('saved_anneal', size)



def kmeans_anneal(X,Y,idx_condensor,name):
    import time
    import pickle
    import numpy as np
    from subset_selection.rec_annealing import anneal_optimize


    cond_anneal = []

    for index in idx_condensor:
        cond_anneal.append(anneal_optimize(index, X, Y))
        pickle.dump(cond_anneal, open('data_sub/'+name+'_kmeans_anneal.p', 'wb'))
        print('saved_kmeans_anneal', len(index))


def random_anneal(X,Y,name,size=100,frac=1.1,tl=60,trial=1):
    import time
    import pickle
    import numpy as np
    from subset_selection.rec_annealing import anneal_optimize

    #X, Y = pickle.load(open('data_embed/mnist_dim32.p', 'rb'))


    cond_anneal = []

    while size < len(Y):

        for i in range(trial):
            index = np.random.choice(list(range(len(Y))), size, replace=False)
            cond_anneal.append(anneal_optimize(index, X, Y, tl))
        pickle.dump(cond_anneal, open('data_sub/'+name+'_random_anneal.p', 'wb'))
        size = int(size * frac)
        print('saved_random_anneal', size)

class Args(object):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("type")
    parser.add_argument("path")
    parser.add_argument("save_path")
    parser.add_argument("final_size", type=int)
    parser.add_argument("throw_frac", type=float)
    args = parser.parse_args()
    print ("cmd line args as follows :")
    print ("type", args.type)
    print ("path", args.path)
    print ("save_path", args.save_path)
    import pickle
    X,Y = pickle.load(open(args.path,'rb'))

    if args.type == 'tiers':
        X_tr_emb, Y_tr = pickle.load(open(args.path, "rb")) 
        condensor_args = Args()
        condensor_args.X_tr_emb = X_tr_emb
        condensor_args.Y_tr = Y_tr
        condensor_args.save_path = args.save_path
        condensor_args.final_size = args.final_size
        condensor_args.throw_frac = args.throw_frac
        condensor_args.require_loss = True
        condensor(condensor_args)
    # if args.type == 'kmeans':
    #     kmeans(X,Y,name)
    #     data_tier_path = 'data_sub/' + name + '_kmeans.p'
    #     index = pickle.load(open(data_tier_path, "rb"))
    #     kmeans_anneal(X, Y, index, name)
    # if args.type == 'random':
    #     random_anneal(X,Y,name)
