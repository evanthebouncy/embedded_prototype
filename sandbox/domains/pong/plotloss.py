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


x1,y1 = get('ppo2_losses')
x2,y2 = get('ppo2_losses_with_training')
x3,y3 = get('ppo2_losses_selected_subset')
x6,y6 = get('ppo2_losses_selected_subset_50000')
x4,y4 = get('ppo2_losses_random_subset')
x5,y5 = get('ppo2_losses_random_projection')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(x1,y1)
ax.plot(x2,y2)
ax.plot(x3,y3)
ax.plot(x4,y4)
ax.plot(x5,y5)
ax.plot(x6,y6)
plt.savefig('finalgg.png')
