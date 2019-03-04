""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
#import torchvision.transforms as T



# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

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

class PG(nn.Module):
    def __init__(self, h, w):
        super(PG, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 10) # 448 or 512
        self.fc2 = nn.Linear(10,1)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        return F.sigmoid(self.fc2(self.fc1(x.view(x.size(0), -1))))

    def get_loss(self, x, y):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.sogmoid(self.fc2(self.fc1(x.view(x.size(0), -1))))
        return torch.nn.MSELoss()(x,y)

    def learn_once(self, inputs, labels):  # inputs = embs,lengths
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels)
        loss.backward()
        self.opt.step()
        return loss

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

def prepro(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    frame = np.expand_dims(frame, -1)
    frame = np.transpose(frame, (2, 0, 1))
    return frame



def observation(self, frame):
    if self.grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    if self.grayscale:
        frame = np.expand_dims(frame, -1)
    return frame

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r




#env = gym.make("Pong-v0")
env = gym.make("PongNoFrameskip-v4")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs, goals = [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

model = PG(84,84)
model.cuda()


def traintheshit(epoch=100,batch_size = 64):
    import pickle
    import random
    path = 'inspected_memory'

    with open(path, 'rb') as f:
        data = pickle.load(f)

    xs = []
    goals = []
    for obs_t, action, reward, obs_tp1, done in data:
        x = obs_t[]-obst[]
        xs.append(x)
        goals.append(1 if action==2 else -1)

    for i in range(epoch):
        print(i)
        idxes = [random.randint(0, len(xs) - 1) for _ in range(batch_size)]
        model.learn_once(to_torch(np.array(xs[idxes]), 'float', True), to_torch(np.array(goals[idxes]), 'float', True))



while True:

    traintheshit()
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    #print(observation.shape)
    cur_x = prepro(observation)
    #print(cur_x.shape)
    x = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob = model(to_torch(np.expand_dims(x, axis=0),'float',True))
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice! big aprob means action == 2

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation

    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(
        y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    goals.append(1 if action==2 else -1)
    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    #print(reward)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        print('done')
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        #epx = np.vstack(xs)
        #eph = np.vstack(hs)
        #epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        #

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        #epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        goals *= discounted_epr

        model.learn_once(to_torch(np.array(xs),'float',True),to_torch(np.array(goals),'float',True))

        xs,goals = [],[]
        xs, hs, dlogps, drs, goals = [], [], [], [], []  # reset array memory

        #grad = policy_backward(eph, epdlogp)
        #for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        #if episode_number % batch_size == -1:
        #    for k, v in model.items():
        #        g = grad_buffer[k]  # gradient
        #        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
        #        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        #        grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 1 == 0:
            torch.save(model.state_dict(), 'model')
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None


    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
