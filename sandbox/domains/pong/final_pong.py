import gym
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from gym import spaces
from collections import deque

import cv2

env = gym.make("PongNoFrameskip-v4")

env.seed(1); torch.manual_seed(1);

learning_rate = 0.0001
gamma = 0.99

if torch.cuda.is_available():
  def to_torch(x, dtype, req = False):
    tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
    return x
else:
  def to_torch(x, dtype, req = False):
    tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
    return x


# state is the difference
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob=prepro(ob)[0]
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(list(self.frames))
env = FrameStack(env,4)

class agent(object):
    def __init__(self,nn):
        self.nn = nn


    def act(self,state,act = None):
        #print(state.shape,'hhh')
        state = np.array(state)
        if state.ndim == 3:
            state = np.array([state])
        state = to_torch(state,'float',True)
        state = self.nn(state)
        #print('shape of state',state.shape)
        c = Categorical(state)
        if act is not None:
            return c
        action = c.sample()
        #print(action)
        # Add log probability of our chosen action to our history
        #if policy.policy_history.dim() != 0:
        #    policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
        #else:
        #    policy.policy_history = (c.log_prob(action))
        return action

    # def learn(self, state, action):
    # do supervised learning on state / action pairs

    # def learn_pg(self, state, action, reward):

def prepro(frame):
    #print(frame.shape)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    frame = np.expand_dims(frame, -1)
    frame = np.transpose(frame, (2, 0, 1))
    return frame



def get_rollout(env,agent,steps=100):
    states = []
    actions = []
    rewards = []
    policy = Variable(torch.Tensor())
    for episode in range(steps):
        state = env.reset()  # Reset environment and record the starting state
        done = False
        i=0
        ps = None
        for time in range(100000):
            i=i+1

            #state = prepro(state)
            #state = state[0]
            if ps is None:
                ps = np.zeros_like(state,float)
            states.append(state)

            action = agent.act(state)
            actions.append(action.data.cpu().numpy()[0])
            ps = state
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.data[0]+1) 
            # 012 -> 123
            #state = prepro(state)


            # Save reward
            rewards.append(reward)
            #print(i,action.data[0],reward,done)
            if reward != 0:
                print(reward)
                break


        # Used to determine when the environment is solved.


        update_reward(rewards,i)
    #rewards = to_torch(np.array(rewards),'float',True)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    return np.array(states),np.array(actions),np.array(rewards) # state is frame diff, action is 0 or 1, rewards are updated rewards

def train(agent,trace,pr=False):
    states,actions,rewards = trace
    states = np.array(states)
    actions = np.array(actions)
    # rewards = to_torch(np.array(rewards),'float',True)
    rewards = to_torch(np.ones(shape=rewards.shape,dtype = float),'float',False)
    c = agent.act(states,actions)
    actions = to_torch(actions,'float',False)
    policy = c.log_prob(actions)
    optimizer.zero_grad()
    loss = -torch.sum(torch.mul(policy, Variable(rewards)))
    # Update network weights
    loss.backward()
    if pr:
        print(loss)
    optimizer.step()

def train_large(agent,trace,epoch=50000,batch_size = 64):
    import random
    states, actions, rewards = trace
    print(len(states))
    for i in range(epoch):
        if i % 100 == 0:
            print(i)
        #idxes = [random.randint(1000, len(states)-100) for _ in range(batch_size)]
        idxes = list(range(40))
        train(agent,(states[idxes],actions[idxes],rewards[idxes]),i%100==0)
        agent.nn.get_loss(states[idxes],actions[idxes],i%100==0)
    idxes = list(range(len(states)-80,len(states)-10))
    idxes = list(range(40))
    agent.nn.get_loss(states[idxes],actions[idxes],True)


def get_stored_trace():
    import pickle
    import random
    path = 'baselines/inspected_large_memory'

    with open(path, 'rb') as f:
        data = pickle.load(f)

    xs = []
    goals = []
    actions = []
    rewards = []

    for obs_t, action, reward, obs_tp1, done in data:
        #print(obs_t.shape) # (84,84,4)
        x = np.transpose(obs_t, (2, 0, 1))
        x = x[0:4]
        #x = np.expand_dims(x,-1)
        #x = np.transpose(x,(2,0,1))
        #print(x.shape)
        xs.append(x)
        if action == 0:
            action = 1
        elif action == 4:
            action = 2
        elif action == 5:
            action = 3
        action = action -1
        actions.append(torch.tensor([action]))
        rewards.append(reward)
        if reward!=0: 
            print(reward)
    update_reward(rewards,0,True)
    return np.array(xs),np.array(actions),np.array(rewards)

def update_reward(reward,steps,all=False):
    if all:
        R = 0
        for i in range(1, len(reward) + 1):
            R = reward[-i] + policy.gamma * R
            if reward[-i]!=0:
                R = reward[-i]
            reward[-i] = R
        return

    R = 0
    for i in range(1,steps+1):
        R = reward[-i] + policy.gamma * R
        reward[-i] = R


class CNN1(nn.Module):

    # init with (channel, height, width) and out_dim for classiication
    def __init__(self, ch_h_w, out_dim):
        super(CNN1, self).__init__()
        self.name = "CNN1"

        self.ch, self.h, self.w = ch_h_w

        self.conv1 = nn.Conv2d(self.ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(6480, 50)
        self.fc2 = nn.Linear(50, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0001) # 0.0001 best for NLL loss

        self.gamma = gamma

    def forward(self, x, log=False):
        #print(x.shape)
        x=self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 6480)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1) if log == False else F.log_softmax(x,dim=1)
    def get_loss(self, X_sub, Y_sub,pr=False):
        X_sub = to_torch(X_sub, "float")
        Y_sub = to_torch(Y_sub, "int")

        # optimize
        output = self.forward(X_sub,log=True)

        loss = F.nll_loss(output, Y_sub)
        if pr:
            print(loss)

        return loss


    def learn_once(self, X_sub, Y_sub,pr=False):
        X_sub = to_torch(X_sub, "float")
        Y_sub = to_torch(Y_sub, "int")

        # optimize
        self.opt.zero_grad()
        output = self.forward(X_sub,log=True)

        loss = F.nll_loss(output, Y_sub)
        loss.backward()
        self.opt.step()
        if pr:
            print(loss)

        return loss

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

policy = CNN1((4,84,84),3)
policy.cuda()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

age = agent(policy)




def update_policy(exp):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


def main():
    trace = get_stored_trace()
    train_large(age,trace)
    return
    while True:
        trace = get_rollout(env,age,100)
        break
        #train(age,trace)
        #trace = get_stored_trace()
        #train_large(age,trace)



main()

