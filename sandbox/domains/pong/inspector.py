import pickle
import numpy as np

path = 'pong_tiers_b.p'

with open(path,'rb') as f:
    data=pickle.load(f)
ans = 0
for i in data[0]:
    print(len(i))
    ans = ans+len(i)
print(ans)
