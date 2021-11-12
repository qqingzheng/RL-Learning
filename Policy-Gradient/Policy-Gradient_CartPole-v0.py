import collections

import gym
import torch
import numpy as np
from torch import nn
import torch.optim
from Core import Utils,DModule,Mpl


class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4,32),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, 2)
        )
        self.softmax = torch.nn.Softmax()
    def forward(self,x):
        x = self.layers.forward(x)
        return self.softmax(x)
EPISODE = 1000
TEST_EPISODE = 20
device = "cpu" if not torch.cuda.is_available() else "cuda"
env = gym.make("CartPole-v0").unwrapped
env.reset()
nn = NN().to(device)
criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.RMSprop(nn.parameters())
pg = DModule.Basic_Policy_Gradient(nn,device,optim,list(range(env.action_space.n)))
vt = Mpl.ViewTrend(0,"Training","EPISODE","STEP_TIMES")
eposide_step = []
for episode in range(EPISODE):
    state = torch.tensor(env.reset()).unsqueeze(0)
    step_times= 0
    while True:
        action = pg.choose_action(state)
        next_state,reward,done,_ = env.step(action)
        pg.learn(state,action,reward)
        state = torch.tensor(next_state).unsqueeze(0)
        env.render()
        if done:
            break
        step_times += 1
    pg.optim_network()
    eposide_step.append(step_times)
    if episode % 20 == 0:
        vt.update(eposide_step)
vt.savefig(eposide_step)
nn.eval()
torch.save(nn,"model_structure.pth")
torch.save(nn.load_state_dict(),"model_weight.pth")

