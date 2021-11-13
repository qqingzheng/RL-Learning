import collections

import gym
import torch
import numpy as np
from torch import nn
import torch.optim
from Core import DModule,Mpl


class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(8,32),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, 4)
        )
        self.softmax = torch.nn.Softmax()
    def forward(self,x):
        x = self.layers.forward(x)
        return self.softmax(x)
EPISODE = 150
TEST_EPISODE = 20
device = "cpu" if not torch.cuda.is_available() else "cuda"
env = gym.make("LunarLander-v2").unwrapped
env.reset()
nn = NN().to(device)
criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.RMSprop(nn.parameters())
pg = DModule.Basic_Policy_Gradient(nn,device,optim,list(range(env.action_space.n)))
vt = Mpl.ViewTrend(0,"Training","EPISODE","STEP_TIMES")
eposide_step = []
avg_eposide_step = []
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
        step_times += reward
    pg.optim_network()
    eposide_step.append(step_times)
    avg_eposide_step.append(np.array(eposide_step).mean())
    if episode % 5 == 0:
        vt.update(eposide_step,avg_eposide_step)
print("done!")
vt.savefig(eposide_step, avg_eposide_step)
nn.eval()
torch.save(nn,"model_structure.pth")
torch.save(nn.load_state_dict(),"model_weight.pth")

