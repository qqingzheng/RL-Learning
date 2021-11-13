import torch.nn
import math
import random
from collections import deque,namedtuple
import numpy as np

"""
DModule.py
"""
class RL_DModule(object):
    def __init__(self, action_list):
        self.step_times = 0
        self.action_list = action_list
    def choose_action(self, **args):
        pass
    def learn(self, **args):
        pass
class DQN(RL_DModule):
    def __init__(self,network,fixed_network,device,optim_fn,loss_fn,action_list
                 ,batch_size=256,memory_size=10000, gamma=0.99
                 , epsilon_start=0.95, epsilon_end=0.1, epsilon_decay=200):
        super(DQN, self).__init__(action_list)
        self.device = device
        self.network = network
        self.fixed_network = fixed_network
        self.Experience = namedtuple("Brain",("state","action","reward","next_state"))
        self.memory = deque([],maxlen=memory_size)
        self.gamma = gamma
        self.optim_times = 0
        self.batch_size = batch_size
        self.optim_fn = optim_fn
        self.loss_fn = loss_fn
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.threshold = 0
    def choose_action_val(self,state):
        with torch.no_grad():
            return self.network(state).item()
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
        action_batch = torch.tensor(next(batch_memory),dtype=torch.int64)
        action_batch = action_batch.reshape((action_batch.shape[0], -1))
        reward_batch = torch.tensor(next(batch_memory), device=self.device)
        next_state_batch_orgin = next(batch_memory)
        not_none_mask = torch.tensor(tuple(map(lambda s: s != None, next_state_batch_orgin)), device=self.device,
                                     dtype=torch.bool)
        next_state_batch = torch.cat([s for s in next_state_batch_orgin if s != None])
        x = self.network(state_batch)
        if len(x.squeeze().shape) == 0:
            pred_state_action = x.gather(1, action_batch)
        else:
            pred_state_action = x
        pred_next_state_action = torch.zeros(state_batch.shape[0])
        with torch.no_grad():
            pred_next_state_action[not_none_mask] = self.fixed_network(next_state_batch).detach().max(1)[0]
        expected_state_action_values = (pred_next_state_action * self.gamma) + reward_batch
        self.optim_fn.zero_grad()
        loss = self.loss_fn(pred_state_action, expected_state_action_values.unsqueeze(1))
        loss.backward()
        self.optim_fn.step()
        self.optim_times += 1
    def update_fixed_network(self):
        self.fixed_network.load_state_dict(self.network.state_dict())
    def learn(self,state,action,reward,next_state):
        experience = self.Experience(state,action,reward,next_state)
        self.memory.append(experience)
class Basic_Policy_Gradient(RL_DModule):
    def __init__(self,network,device,optim_fn,action_list,lr=1e-2,gamma=0.95):
        super(Basic_Policy_Gradient, self).__init__(action_list)
        self.device = device
        self.network = network
        self.ep_action,self.ep_reward,self.ep_state = [],[],[]
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
        returns = torch.tensor((returns - returns.mean())/(returns.std() + 1e-5), device=self.device,requires_grad=False)
        returns = returns.reshape(pred_state_action.shape[0],1)
        self.optim_fn.zero_grad()
        action_mask = torch.zeros_like(pred_state_action,requires_grad=False)
        for i in range(len(ep_action)):
            action_mask[i,ep_action[i]] = 1
        loss = -(torch.log(pred_state_action)*returns*action_mask).sum()/pred_state_action.shape[0]
        loss.backward()
        self.optim_fn.step()
        self.ep_action, self.ep_reward, self.ep_state = [], [], []
    def choose_action(self,state):
        with torch.no_grad():
            pred_action = self.network(state).detach()
            pred_action = pred_action.squeeze().numpy()

            return np.random.choice(self.action_list,p=pred_action)
class Actor_Critic():
    def __init__(self,actor_network,critic_network,critic_fixed_network,device,actor_optim_fn,critic_optim_fn
                 ,action_list,actor_lr=1e-3,critic_lr=1e-2,actor_gamma=0.95
                 ,batch_size=128, memory_size=40000, critic_gamma=0.99
                 ):
        self.device = device
        self.actor_network = actor_network
        self.actor_optim_fn = actor_optim_fn
        self.actor = Basic_Policy_Gradient(actor_network,device,actor_optim_fn,action_list,lr=actor_lr,gamma=actor_gamma)
        self.critic_network = critic_network
        self.critic_optim_fn = critic_optim_fn
        self.critic_fixed_network = critic_fixed_network
        self.critic_loss_fn = torch.nn.MSELoss()
        self.critic = DQN(critic_network,critic_fixed_network,device,critic_optim_fn,self.critic_loss_fn,action_list
                 ,batch_size, memory_size, gamma=critic_gamma)
        self.action_list = action_list
        self.batch_size = batch_size
        self.critic_gamma = critic_gamma
        self.Experience = namedtuple("Brain", ("state", "action", "reward", "next_state"))
        self.memory = deque([], maxlen=memory_size)
        self.last_reward = 0
    def critic_learn(self,state,action,reward,next_state):
        self.critic.learn(state, action, reward,next_state)
        self.optim_critic()
    def actor_learn(self,state,action,td_error):
        self.actor.learn(state,action,td_error)
    def optim_critic(self):
        self.critic.optim_network()
    def optim_actor(self):
        self.actor.optim_network()
    def get_error(self,state):
        return self.critic_network(state)
    def choose_action(self,state):
        return self.actor.choose_action(state)
    def update_critic_fixed_network(self):
        self.critic_fixed_network.load_state_dict(self.critic_network.state_dict())
class DDPG():
    def __init__(self,actor_network,actor_fixed_network,critic_network,critic_fixed_network,device,actor_optim_fn,critic_optim_fn
                 ,action_list,actor_lr=1e-3,actor_gamma=0.95
                 ,batch_size=128, memory_size=40000, critic_gamma=0.99
                 , epsilon_start=0.95, epsilon_end=0.1, epsilon_decay=200
                 ):
        self.device = device
        self.actor_network = actor_network
        self.actor_fixed_network = actor_fixed_network
        self.actor_optim_fn = actor_optim_fn
        self.actor = Basic_Policy_Gradient(actor_network,device,actor_optim_fn,action_list,lr=actor_lr,gamma=actor_gamma)
        self.actor_target = Basic_Policy_Gradient(actor_network, device, actor_optim_fn, action_list, lr=actor_lr,
                                           gamma=actor_gamma)
        self.critic_network = critic_network
        self.critic_optim_fn = critic_optim_fn
        self.critic_fixed_network = critic_fixed_network
        self.critic_loss_fn = torch.nn.MSELoss()
        self.critic = DQN(critic_network,self.critic_fixed_network,device,critic_optim_fn,self.critic_loss_fn,action_list
                 ,batch_size, memory_size, critic_gamma, epsilon_start, epsilon_end, epsilon_decay)
        self.action_list = action_list
        self.batch_size = batch_size
        self.critic_gamma = critic_gamma
        self.Experience = namedtuple("Brain", ("state", "action", "reward", "next_state"))
        self.memory = deque([], maxlen=memory_size)
        self.steps = 0
    def critic_learn(self,state,action,reward,next_state):
        self.critic.learn(state,action,reward,next_state)
        self.optim_critic()
    def actor_learn(self,state,action,td_error):
        self.actor.learn(state,action,td_error)
    def optim_critic(self):
        self.critic.optim_network()
    def optim_actor(self):
        self.actor.optim_network()
    def get_error(self,state):
        return self.critic_network(state)
    def choose_action(self,state):
        return self.actor_target.choose_action(state)
    def update_fixed_network(self):
        self.actor_fixed_network.load_state_dict(self.actor_network.state_dict())
        self.critic_fixed_network.load_state_dict(self.critic_network.state_dict())

# class Actor_Critic():
#     def __init__(self,actor_network,critic_network,critic_fixed_network,device,actor_optim_fn,critic_optim_fn
#                  ,action_list,actor_lr=1e-3,critic_lr=1e-2,actor_gamma=0.95
#                  ,batch_size=128, memory_size=40000, critic_gamma=0.99
#                  ):
#         self.device = device
#         self.actor_network = actor_network
#         self.actor_optim_fn = actor_optim_fn
#         self.actor = Basic_Policy_Gradient(actor_network,device,actor_optim_fn,action_list,lr=actor_lr,gamma=actor_gamma)
#         self.critic_network = critic_network
#         self.critic_optim_fn = critic_optim_fn
#         self.critic_fixed_network = critic_fixed_network
#         self.action_list = action_list
#         self.batch_size = batch_size
#         self.critic_gamma = critic_gamma
#         self.Experience = namedtuple("Brain", ("state", "action", "reward", "next_state"))
#         self.memory = deque([], maxlen=memory_size)
#         self.last_reward = 0
#     def get_batch_from_memory(self):
#         return random.sample(self.memory, self.batch_size)
#     def critic_learn(self,state,action,reward,next_state):
#         experience = self.Experience(state, action, reward,next_state)
#         self.memory.append(experience)
#         self.optim_critic()
#     def actor_learn(self,state,action,td_error):
#         self.actor.learn(state,action,td_error)
#     def optim_critic(self):
#         if len(self.memory) < self.batch_size:
#             return
#         zip_memory = zip(*self.get_batch_from_memory())
#         state_batch = torch.cat(next(zip_memory))
#         action_batch = torch.tensor(next(zip_memory))
#         reward_batch = torch.tensor(next(zip_memory))
#         next_state_batch_orgin = next(zip_memory)
#         not_none_mask = torch.tensor(tuple(map(lambda s: s != None, next_state_batch_orgin)), device=self.device,
#                                      dtype=torch.bool)
#         next_state_batch = torch.cat([s for s in next_state_batch_orgin if s != None])
#         td_error_batch = self.critic_network(state_batch)
#         next_td_error_no_mask = self.critic_fixed_network(next_state_batch).detach()
#         next_td_error = torch.zeros(state_batch.shape[0])
#         with torch.no_grad():
#             next_td_error[not_none_mask] = next_td_error_no_mask[:,0]
#         expected = reward_batch + self.critic_gamma*(next_td_error)
#         loss_fn = torch.nn.MSELoss()
#         td_error_batch = td_error_batch.type(torch.float32)
#         expected = expected.type(torch.float32)
#         self.critic_optim_fn.zero_grad()
#         loss = loss_fn(td_error_batch,expected.unsqueeze(1))
#         print(f"\r loss: {loss}", end="")
#         loss.backward()
#         self.critic_optim_fn.step()
#     def optim_actor(self):
#         self.actor.optim_network()
#     def get_error(self,state):
#         return self.critic_network(state)
#     def choose_action(self,state):
#         return self.actor.choose_action(state)
#     def update_critic_fixed_network(self):
#         self.critic_fixed_network.load_state_dict(self.critic_network.state_dict())