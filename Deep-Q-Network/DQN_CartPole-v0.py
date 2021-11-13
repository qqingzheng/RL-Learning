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
            torch.nn.Linear(4,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,2)
        )
    def forward(self,x):
        return self.layers.forward(x)
EPISODE = 20000
device = "cpu" if not torch.cuda.is_available() else "cuda"
env = gym.make("CartPole-v0").unwrapped
env.reset()
nn = NN().to(device)
fixed_nn = NN().to(device)
fixed_nn.load_state_dict(nn.state_dict())
fixed_nn.eval()
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(nn.parameters(),lr=5e-3)
dqn = DModule.DQN(nn,fixed_nn,device,optim,criterion,list(range(env.action_space.n)),epsilon_end=0.05,epsilon_decay=200,gamma=0.8)
vt = Mpl.ViewTrend(0,"Training","EPISODE","STEP_TIMES")
eposide_step = []
for episode in range(EPISODE):
    state = torch.tensor(env.reset()).unsqueeze(0)
    step_times= 0
    while True:
        action = dqn.choose_action(state)
        next_state,reward,done,_ = env.step(action)
        if not done:
            next_state = torch.tensor(next_state).unsqueeze(0)
        else:
            next_state = None
        dqn.learn(state,action,reward,next_state)
        state = next_state
        dqn.optim_network()
        env.render()
        step_times += 1
        if done:
            break
    eposide_step.append(step_times)
    if episode % 5 == 0:
        dqn.update_fixed_network()
    if episode % 10 == 0:
        vt.update(eposide_step)
print("done!")
vt.savefig(eposide_step)
fixed_nn.eval()
torch.save(fixed_nn, 'model_structure.pth')
torch.save(fixed_nn.state_dict(), 'model_weights.pth')