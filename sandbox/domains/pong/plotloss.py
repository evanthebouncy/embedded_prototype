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

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(x1,y1)
ax.plot(x2,y2)
plt.show()