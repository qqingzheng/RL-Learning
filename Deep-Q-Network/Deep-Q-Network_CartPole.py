import random
import math
import gym
import torch
from collections import deque,namedtuple
import numpy as np
import matplotlib
from torch import nn
import torch.optim
from IPython import display
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from Core import Utils
plt.ion()
def update_view(x):
    plt.figure(2)
    plt.clf()
    plt.title('Screen sliced view')
    plt.imshow(x)
    plt.pause(0.001)
    display.clear_output(wait=True)
    display.display(plt.gcf())
def get_screen():
    screen = np.pad(env.render(mode='rgb_array').transpose((2, 0, 1)),((0,0),(0,0),(100,100)),mode='constant')
    cart_position = int((2.4 + env.state[0])*(800/4.8))
    screen = screen[:,240:270,(cart_position-40):(cart_position+40)]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    ip = Utils.ImgPreproccess(30)
    #update_view(ip(screen).permute((1,2,0)))
    return ip(screen).unsqueeze(0)
class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten()
        )

        self.leaner = nn.Linear(224,2)
    def forward(self,x):
        x = self.layers.forward(x)
        return self.leaner.forward(x)
Record = namedtuple("Record",["state","action","reward","next_state"])
class Memory():
    def __init__(self,len):
        self.memory = deque([],maxlen=len)
    def push(self,x):
        self.memory.append(x)
    def get(self,batch_size):
        return random.sample(self.memory,batch_size)
env = gym.make("CartPole-v0").unwrapped
device = "cpu" if not torch.cuda.is_available() else "cuda"
EPISODES = 1000
EPSILON_START = 1
EPSILON_END = 0.05
EPSILON_DECAY = 800
GAMMA = 0.99
BATCH_SIZE = 64
step_times = 0
Net = NN().to(device)
fixedNet = NN().to(device)
fixedNet.load_state_dict(Net.state_dict())
optim = torch.optim.RMSprop(Net.parameters())
memory = Memory(40000)
def get_action(state):
    global step_times
    sample = torch.rand(1).item()
    threshold = EPSILON_END + (EPSILON_START - EPSILON_END)*math.exp(-1. * step_times / EPSILON_DECAY)
    step_times += 1
    if sample < threshold:
        return np.random.choice((0,1))
    else:
        with torch.no_grad():
            return Net.forward(state).data.max(1)[1].item()
def train():
    if len(memory.memory) < BATCH_SIZE:
        return
    batch_memory = zip(*memory.get(BATCH_SIZE))
    state_batch = torch.cat(next(batch_memory))
    action_batch = torch.tensor(next(batch_memory))
    action_batch = action_batch.reshape((action_batch.shape[0],-1))
    reward_batch = torch.tensor(next(batch_memory),device=device)
    next_state_batch_orgin = next(batch_memory)
    not_none_mask = torch.tensor(tuple(map(lambda s: s != None,next_state_batch_orgin)),device=device,dtype=torch.bool)
    next_state_batch = torch.cat([s for s in next_state_batch_orgin if s != None])
    pred_state_action = Net(state_batch).gather(1,action_batch)
    # pred_state_action shape(batch_size,n_action)
    pred_next_state_action = torch.zeros(state_batch.shape[0])
    with torch.no_grad():
        pred_next_state_action[not_none_mask] = fixedNet(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (pred_next_state_action * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    optim.zero_grad()
    loss = criterion(pred_state_action,expected_state_action_values.unsqueeze(1))
    loss.backward()
    optim.step()
for episode in range(EPISODES):
    env.reset()
    times = 0
    former = get_screen()
    latter = get_screen()
    state = former - latter
    while True:
        times+=1
        action = get_action(state)
        _,reward,is_done,_ = env.step(action)
        former = latter
        latter = get_screen()
        next_state = former - latter if not is_done else None
        memory.push(Record(state,action,reward,next_state))
        state = next_state
        train()
        if is_done:
            print(f"Episode {episode} -> duration:{times}")
            break
    if (episode % 3 == 0):
        print("Fixed Q target has updated!")
        fixedNet.load_state_dict(Net.state_dict())