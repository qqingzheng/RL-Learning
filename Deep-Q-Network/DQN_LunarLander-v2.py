import gym
import torch
import numpy as np
from torch import nn
import torch.optim
from Core import Utils,DModule,Mpl
class NN(nn.Module):
    def __init__(self,n_states,n_actions):
        super(NN,self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_states,32),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32,n_actions)
        )
    def forward(self,x):
        return self.layers.forward(x)
EPISODE = 200
TEST_EPISODE = 20
device = "cpu" if not torch.cuda.is_available() else "cuda"
env = gym.make("LunarLander-v2").unwrapped
env.reset()
nn = NN(env.observation_space.shape[0],env.action_space.n).to(device)
fixed_nn = NN(env.observation_space.shape[0],env.action_space.n).to(device)
fixed_nn.load_state_dict(nn.state_dict())
criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.RMSprop(nn.parameters())
dqn = DModule.DQN(nn,fixed_nn,device,optim,criterion,list(range(env.action_space.n)),epsilon_end=0.05,epsilon_decay=200,gamma=0.999,memory_update_step=10)
vt = Mpl.ViewTrend(0,"Training","EPISODE","STEP_TIMES")
eposide_step = []
avg_eposide_step = []
for episode in range(EPISODE):
    state = torch.tensor(env.reset()).unsqueeze(0)
    total_reward = 0
    while True:
        action = dqn.choose_action(state,is_test=False if episode%TEST_EPISODE != 0 else True)
        next_state,reward,done,_ = env.step(action)
        next_state = torch.tensor(next_state).unsqueeze(0)
        dqn.learn(state,action,reward,next_state)
        state = next_state
        dqn.optim_network()
        env.render()
        total_reward += reward
        if done:
            break
    eposide_step.append(total_reward)
    avg_eposide_step.append(np.array(eposide_step).mean())
    if episode % 3 == 0:
        vt.update(eposide_step, avg_eposide_step)
print("done!")
vt.update(eposide_step, avg_eposide_step)
fixed_nn.eval()
torch.save(fixed_nn, 'model_structure.pth')
torch.save(fixed_nn.state_dict(), 'model_weights.pth')