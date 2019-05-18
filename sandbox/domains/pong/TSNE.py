import numpy as np
import pickle
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize(X,Y,graph_path):
    X_embedded = TSNE(n_components=2).fit_transform(X)
    y=['red' if Y[i]==0 else 'blue' for i in range(Y.shape[0])]
    plt.scatter(X_embedded[0],X_embedded[1],c=y)
    plt.savefig(graph_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("graph_path")
    parser.add_argument("idx_path")
    parser.add_argument("mem_path")

    args = parser.parse_args()

    with open(args.idx_path,"rb") as f:
        idx = pickle.load(f)
    X,Y = pickle.load(open(args.mem_path,'rb'))
    X,Y=X[idx],Y[idx]
    visualize(X,Y,args.graph_path)



