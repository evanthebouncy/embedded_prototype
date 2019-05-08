import numpy as np
import pickle
import argparse

def generate_random_pickle(tot_size,size,path):
    inds = np.arange(tot_size)

    np.random.shuffle(inds)
    inds = inds[:size]
    pickle.dump(inds, open(path, "wb"))

def generate_subset(tot_size,size,path,save_path):

    with open(path,'rb') as f:
        idx = pickle.load(f)
    rm = []





    tot = tot_size
    for i in range(len(idx[0])-1,0,-1):
        print(len(idx[0][i]))
        rm.extend(idx[0][i])
        if tot-len(rm)<size:
            break
    inds = np.arange(tot).tolist()
    rm=set(rm)
    inds = [i for i in inds if i not in rm]

    inds = np.array(inds)
    pickle.dump(inds, open(save_path, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tot_size")
    parser.add_argument("type")
    parser.add_argument("size")
    parser.add_argument("save_path")
    parser.add_argument("load_path")

    args = parser.parse_args()
    print ("cmd line args as follows :")
    print("type",args.type)
    print ("tot_size", args.tot_size)
    print ("size", args.size)
    print ("save_path", args.save_path)


    if args.type == 'random':
        generate_random_pickle(args.tot_size,args.size,args.save_path)
    if args.type == 'subset':
        generate_subset(args.tot_size,args.size,args.load_path,args.save_path)
    if args.type == 'all':
        generate_random_pickle(args.tot_size, args.tot_size, args.save_path+'_whole.p')
        generate_random_pickle(args.tot_size, args.size, args.save_path + '_random.p')
        generate_subset(args.tot_size, args.size, args.load_path, args.save_path+'_subset.p')



