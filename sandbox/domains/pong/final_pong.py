import gym
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import cv2

env = gym.make("PongNoFrameskip-v4")
env.seed(1); torch.manual_seed(1);

learning_rate = 0.01
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

            state = prepro(state)
            if ps is None:
                ps = np.zeros_like(state,float)

            states.append(state-ps)

            action = agent.act(state-ps)
            actions.append(action.data[0])
            ps = state
            # Step through environment using chosen action
            state, reward, done, _ = env.step(2 if action.data[0]==0 else 3)
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

def train(agent,trace):
    states,actions,rewards = trace
    states = np.array(states)
    actions = np.array(actions)
    rewards = to_torch(np.array(rewards),'float',True)
    c = agent.act(states,actions)
    actions = to_torch(actions,'float',True)
    policy = c.log_prob(actions)
    loss = (torch.sum(torch.mul(policy, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_large(agent,trace,epoch=1000,batch_size = 64):
    import random
    states, actions, rewards = trace
    for i in range(epoch):
        print(i)
        idxes = [random.randint(0, len(states)-1) for _ in range(batch_size)]
        train(agent,(states[idxes],actions[idxes],rewards[idxes]))


def get_stored_trace():
    import pickle
    import random
    path = 'baselines/inspected_memory'

    with open(path, 'rb') as f:
        data = pickle.load(f)

    xs = []
    goals = []
    actions = []
    rewards = []

    for obs_t, action, reward, obs_tp1, done in data:
        #print(obs_t.shape) # (84,84,4)
        x = np.transpose(obs_t, (2, 0, 1))
        x = x[3]-x[2]
        x = np.expand_dims(x,-1)
        x = np.transpose(x,(2,0,1))
        xs.append(x)
        actions.append(torch.tensor([0]) if action == 2 or action == 4 else torch.tensor([1]))
        rewards.append(reward)
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
    def __init__(self, ch_h_w, out_dim, stop_criteria=(0.01, 1000, 120)):
        super(CNN1, self).__init__()
        self.name = "CNN1"

        self.ch, self.h, self.w = ch_h_w

        self.conv1 = nn.Conv2d(self.ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(6480, 50)
        self.fc2 = nn.Linear(50, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

        self.stop_criteria = stop_criteria
        self.gamma = gamma

    def forward(self, x):
        #print(x.shape)
        x=self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 6480)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

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

policy = CNN1((1,84,84),2)
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
    #trace = get_stored_trace()
    #train_large(age,trace)
    while True:
        trace = get_rollout(env,age,10)
        train(age,trace)



main()

