import torch.nn
import math
import random
from collections import deque,namedtuple
import numpy as np
class RL_DModule(object):
    def __init__(self, action_list):
        self.step_times = 0
        self.action_list = action_list
    def choose_action(self, **args):
        pass
    def learn(self, **args):
        pass
class DQN(RL_DModule):
    def __init__(self,network,fixed_network,device,optim_fn,loss_fn,action_list,batch_size=128,memory_size=40000, gamma=0.99, epsilon_start=0.95, epsilon_end=0.1, epsilon_decay=200, memory_update_step=3):
        super(DQN, self).__init__(action_list)
        self.device = device
        self.network = network
        self.fixed_network = fixed_network
        self.Experience = namedtuple("Brain",("state","action","reward","next_state"))
        self.memory = deque([],maxlen=memory_size)
        self.gamma = gamma
        self.memory_update_step = memory_update_step
        self.optim_times = 0
        self.batch_size = batch_size
        self.optim_fn = optim_fn
        self.loss_fn = loss_fn
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    def choose_action(self, state, is_test=False):
        random_rate = np.random.uniform(0, 1)
        self.threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1 * (self.step_times) / (self.epsilon_decay))
        self.step_times += 1
        if random_rate > self.threshold or is_test:
            with torch.no_grad():
                return self.network(state).argmax().item()
        else:
            return np.random.choice(self.action_list)
    def get_batch_from_memory(self):
        return random.sample(self.memory, self.batch_size)
    def optim_network(self):
        if self.batch_size > len(self.memory):
            return
        batch_memory = zip(*self.get_batch_from_memory())
        state_batch = torch.cat(next(batch_memory))
        action_batch = torch.tensor(next(batch_memory))
        action_batch = action_batch.reshape((action_batch.shape[0], -1))
        reward_batch = torch.tensor(next(batch_memory), device=self.device)
        next_state_batch_orgin = next(batch_memory)
        not_none_mask = torch.tensor(tuple(map(lambda s: s != None, next_state_batch_orgin)), device=self.device,
                                     dtype=torch.bool)
        next_state_batch = torch.cat([s for s in next_state_batch_orgin if s != None])
        pred_state_action = self.network(state_batch).gather(1, action_batch)
        # pred_state_action shape(batch_size,n_action)
        pred_next_state_action = torch.zeros(state_batch.shape[0])
        with torch.no_grad():
            pred_next_state_action[not_none_mask] = self.fixed_network(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (pred_next_state_action * self.gamma) + reward_batch
        self.optim_fn.zero_grad()
        loss = self.loss_fn(pred_state_action, expected_state_action_values.unsqueeze(1))
        loss.backward()
        self.optim_fn.step()
        self.optim_times += 1
        if self.optim_times % self.memory_update_step == 0:
            self.fixed_network.load_state_dict(self.network.state_dict())
    def learn(self,state,action,reward,next_state):
        experience = self.Experience(state,action,reward,next_state)
        self.memory.append(experience)
class Basic_Policy_Gradient(RL_DModule):
    def __init__(self,network,device,optim_fn,action_list,lr=1e-1,gamma=0.95):
        super(Basic_Policy_Gradient, self).__init__(action_list)
        self.device = device
        self.network = network
        self.ep_action,self.ep_reward,self.ep_state = [],[],[]
        self.memory = list()
        self.gamma = gamma
        self.optim_fn = optim_fn
        self.last_reward = 0
    def learn(self,state,action,reward):
        self.ep_state.append(state)
        self.ep_action.append(action)
        self.ep_reward.append(reward)
    def optim_network(self):
        state_batch = torch.cat(self.ep_state)
        ep_action = self.ep_action
        reward_batch = np.array(self.ep_reward,dtype=np.int32)
        self.last_reward = reward_batch.sum()
        returns = np.zeros_like(reward_batch,dtype=np.float64)
        temp = 0.00
        for i in reversed(range(len(reward_batch))):
            temp = temp*self.gamma+reward_batch[i]
            returns[i] = temp
        pred_state_action = self.network(state_batch)
        returns = torch.tensor(returns - returns.mean() / (returns.std() + 1e-9), device=self.device,requires_grad=False)
        returns = returns.reshape(pred_state_action.shape[0],1)
        self.optim_fn.zero_grad()
        action_mask = torch.zeros_like(pred_state_action,requires_grad=False)
        for i in range(len(ep_action)):
            action_mask[i,ep_action[i]] = 1
        loss = (-torch.log(pred_state_action)*returns*action_mask).sum()
        loss.backward()
        self.optim_fn.step()
        self.ep_action, self.ep_reward, self.ep_state = [], [], []
    def choose_action(self,state):
        with torch.no_grad():
            pred_action = self.network(state).detach()
            if torch.isnan(pred_action)[0,0]:
                pred_action = [0.5,0.5]
            else:
                pred_action = pred_action.squeeze().numpy()
            return np.random.choice(self.action_list,p=pred_action)