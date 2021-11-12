import collections

import gym
import torch
import numpy as np
from torch import nn
import torch.optim
from Core import Utils,DModule,Mpl


def get_screen() -> torch.Tensor:
    screen = np.pad(env.render(mode='rgb_array').transpose((2, 0, 1)),((0,0),(0,0),(200,200)),mode='edge')
    cart_position = int((2.4 + env.state[0])*(1000/4.8))
    screen = screen[:,150:300,(cart_position-80):(cart_position+80)]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    ip = Utils.ImgPreproccess(60)
    return ip(screen).unsqueeze(0)
class NN(nn.Module):
    def __init__(self, screen_height, screen_width, action_num):
        super(NN,self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3,affine=False),
            torch.nn.Conv2d(3,16,kernel_size=5,stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16,32,kernel_size=5,stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Flatten()
        )
        window = self.conv_workpace(*self.conv_workpace(screen_width,screen_height,5,2),5,2)
        window_size = window[0]*window[1]
        self.batch1 = torch.nn.BatchNorm2d(16)
        self.batch2 = torch.nn.BatchNorm2d(32)
        self.conv1 = torch.nn.Conv2d(3,16,kernel_size=5,stride=2)
        self.conv2 = torch.nn.Conv2d(16,32, kernel_size=5, stride=2)
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(32*window_size,2)
        self.softmax = torch.nn.Softmax()
        self.flatten = torch.nn.Flatten()
    def conv_workpace(self,weight,height,kernel_size,stride):
        return (weight - kernel_size)//stride + 1,(height - kernel_size)//stride + 1
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batch2(x)
        x = self.flatten(x)
        x = self.linear1.forward(x)
        return self.softmax(x)
EPISODE = 1000
TEST_EPISODE = 20
device = "cpu" if not torch.cuda.is_available() else "cuda"
env = gym.make("CartPole-v0").unwrapped
env.reset()
preview = get_screen()
nn = NN(preview.shape[2],preview.shape[3],env.action_space.n).to(device)
criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.RMSprop(nn.parameters())
pg = DModule.Basic_Policy_Gradient(nn,device,optim,list(range(env.action_space.n)))
vt = Mpl.ViewTrend(0,"Training","EPISODE","STEP_TIMES")
eposide_step = []
for episode in range(EPISODE):
    env.reset()
    observation = get_screen()
    next_observation = get_screen()
    state = next_observation - observation
    step_times= 0
    while True:
        action = pg.choose_action(state)
        _,reward,done,_ = env.step(action)
        pg.learn(state,action,reward)
        observation = next_observation
        next_observation = get_screen()
        state = next_observation - observation
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

