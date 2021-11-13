import collections

import gym
import torch
import numpy as np
from torch import nn
import torch.optim
from Core import Utils,DModule,Mpl


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet,self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4,32),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, 2)
        )
        self.softmax = torch.nn.Softmax()
    def forward(self,x):
        x = self.layers.forward(x)
        return self.softmax(x)
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet,self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4,32),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, 1),
        )
    def forward(self,x):
        return self.layers.forward(x)
EPISODE = 2000
TEST_EPISODE = 20
device = "cpu" if not torch.cuda.is_available() else "cuda"
env = gym.make("CartPole-v0").unwrapped
env.reset()
actor_nn = ActorNet().to(device)
actor_fixed_nn = ActorNet().to(device)
actor_fixed_nn.load_state_dict(actor_nn.state_dict())
actor_fixed_nn.eval()
critic_nn = CriticNet().to(device)
critic_fixed_nn = CriticNet().to(device)
critic_fixed_nn.load_state_dict(critic_nn.state_dict())
critic_fixed_nn.eval()
actor_optim = torch.optim.Adam(actor_nn.parameters(),lr=0.01)
critic_optim = torch.optim.Adam(critic_nn.parameters(),lr=0.01)
ddpg = DModule.DDPG(actor_nn,actor_fixed_nn,critic_nn,critic_fixed_nn,device,actor_optim,critic_optim,list(range(2)),epsilon_decay=400)
vt = Mpl.ViewTrend(0,"Training","EPISODE","STEP_TIMES")
eposide_step = []
avg_eposide_step = []
for episode in range(EPISODE):
    state = torch.tensor(env.reset()).unsqueeze(0)
    step_times= 0
    while True:
        action = ddpg.choose_action(state)
        next_state,reward,done,_ = env.step(action)
        td_error = ddpg.get_error(state).item()
        next_state = torch.tensor(next_state).unsqueeze(0)
        ddpg.critic_learn(state,action,reward,next_state)
        ddpg.actor_learn(state,action,td_error)
        ddpg.optim_critic()
        state = next_state
        env.render()
        if done:
            break
        step_times += 1
    ddpg.optim_actor()
    eposide_step.append(step_times)
    avg_eposide_step.append(np.array(eposide_step).mean())
    if episode % 10 == 0:
        ddpg.update_fixed_network()
        vt.update(eposide_step, avg_eposide_step)
vt.savefig(eposide_step, avg_eposide_step)

