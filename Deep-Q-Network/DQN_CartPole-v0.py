import gym
import torch
import numpy as np
from torch import nn
import torch.optim
from Core import Utils,DModule,Mpl


def get_screen() -> torch.Tensor:
    screen = np.pad(env.render(mode='rgb_array').transpose((2, 0, 1)),((0,0),(0,0),(200,200)),mode='constant')
    cart_position = int((2.4 + env.state[0])*(800/4.8))
    screen = screen[:,150:280,(cart_position-50):(cart_position+50)]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    ip = Utils.ImgPreproccess(30)
    return ip(screen).unsqueeze(0)
class NN(nn.Module):
    def __init__(self, screen_height, screen_width, action_num):
        super(NN,self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,kernel_size=5,stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16,32,kernel_size=5,stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Flatten()
        )
        window = self.conv_workpace(*self.conv_workpace(*self.conv_workpace(screen_width,screen_height,5,2),5,2),5,2)
        window_size = window[0]*window[1]
        self.leaner = torch.nn.Linear(32*window_size,action_num)
    def conv_workpace(self,weight,height,kernel_size,stride):
        return (weight - kernel_size)//stride + 1,(height - kernel_size)//stride + 1
    def forward(self,x):
        x = self.layers.forward(x)
        return self.leaner.forward(x)
EPISODE = 1000
device = "cpu" if not torch.cuda.is_available() else "cuda"
env = gym.make("CartPole-v0").unwrapped
env.reset()
preview = get_screen()
nn = NN(preview.shape[2],preview.shape[3],env.action_space.n).to(device)
fixed_nn = NN(preview.shape[2],preview.shape[3],env.action_space.n).to(device)
fixed_nn.load_state_dict(nn.state_dict())

criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.RMSprop(nn.parameters())
dqn = DModule.DQN(nn,fixed_nn,device,optim,criterion,list(range(env.action_space.n)))
vt = Mpl.ViewTrend(0,"Training","EPISODE","STEP_TIMES")
eposide_step = []
for episode in range(EPISODE):
    env.reset()
    observation = get_screen()
    next_observation = get_screen()
    state = next_observation - observation
    step_times= 0
    while True:
        action = dqn.choose_action(state)
        _,reward,done,_ = env.step(action)
        next_observation = get_screen()
        next_state = next_observation - observation
        dqn.learn(state,action,reward,next_state)
        dqn.optim_network()
        state = next_state
        observation = next_observation
        env.render()
        if done:
            break
        step_times += 1
    eposide_step.append(step_times)
    if episode % 3 == 0:
        vt.update(eposide_step)




