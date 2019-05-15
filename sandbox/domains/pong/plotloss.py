import matplotlib.pyplot as plt
import numpy as np
import pickle

def get(path):
    x=[]
    y=[]

    with open(path,'rb') as f:
        a = pickle.load(f)
    for i in a:
        i1,i2=i
        x.append(i1)
        y.append(i2)
    return x,y


x1,y1 = get('cart_data/result_loss_idx_whole')
x2,y2 = get('cart_data/result_loss_idx_random')
x3,y3 = get('cart_data/result_loss_idx_subset')
#x6,y6 = get('ppo2_losses_selected_subset_50000')
#x4,y4 = get('ppo2_losses_random_subset')
#x5,y5 = get('ppo2_losses_random_projection')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(x1,y1,label='whole')
ax.plot(x2,y2,label='random')
ax.plot(x3,y3,label='subset')

plt.savefig('finalgg.png')
