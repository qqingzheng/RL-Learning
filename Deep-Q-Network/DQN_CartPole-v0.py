import gym
import torch
import numpy as np
from torch import nn
import torch.optim
from Core import Utils,DModule,Mpl
def get_screen() -> torch.Tensor:
    extend = 400
    screen = np.pad(env.render(mode='rgb_array').transpose((2, 0, 1)),((0,0),(0,0),(extend,extend)),mode='edge')
    cart_position = (2.4 + env.state[0])*((600+extend*2)/4.8)
    screen = screen[:,120:400,int((cart_position-300)):int((cart_position+300))]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    ip = Utils.ImgPreproccess(40)
    #iv.update(ip(screen).unsqueeze(0))
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
        self.leaner1 = torch.nn.Linear(32*window_size,2)
    def conv_workpace(self,weight,height,kernel_size,stride):
        return (weight - kernel_size)//stride + 1,(height - kernel_size)//stride + 1
    def forward(self,x):
        x = self.layers.forward(x)
        return self.leaner1.forward(x)
EPISODE = 200
TEST_EPISODE = 20
device = "cpu" if not torch.cuda.is_available() else "cuda"
env = gym.make("CartPole-v0").unwrapped
env.reset()
preview = get_screen()
nn = NN(preview.shape[2],preview.shape[3],env.action_space.n).to(device)
fixed_nn = NN(preview.shape[2],preview.shape[3],env.action_space.n).to(device)
fixed_nn.load_state_dict(nn.state_dict())
criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.RMSprop(nn.parameters())
dqn = DModule.DQN(nn,fixed_nn,device,optim,criterion,list(range(env.action_space.n)),epsilon_end=0.05,epsilon_decay=200,gamma=0.999,memory_update_step=10)
vt = Mpl.ViewTrend(0,"Training","EPISODE","STEP_TIMES")
eposide_step = []
test_eposide_step = []
for episode in range(EPISODE):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    step_times= 0
    while True:
        action = dqn.choose_action(state,is_test=False if episode%TEST_EPISODE != 0 else True)
        _,reward,done,_ = env.step(action)
        next_observation = get_screen()
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
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
    if episode % 10 == 0:
        vt.update(eposide_step)
print("done!")
vt.savefig(eposide_step)
fixed_nn.eval()
torch.save(fixed_nn, 'model_structure.pth')
torch.save(fixed_nn.state_dict(), 'model_weights.pth')