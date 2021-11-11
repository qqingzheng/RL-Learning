import torch
import numpy as np
import math
import pandas as pd
class RL_Module(object):
    def __init__(self,action_list,lr=1e-2,epsilon_start=0.9,epsilon_end=0.1,epsilon_decay=200):
        self.lr = lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_times = 0
        self.action_list = action_list
    def add_state(self, state):
        pass
    def choose_action(self,state,is_test=False):
        self.add_state(state)
        random_rate = np.random.uniform(0,1)
        self.threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end)*math.exp(-1*(self.step_times)/(self.epsilon_decay))
        self.step_times += 1
        if random_rate > self.threshold or is_test:
            return self.action_fn(state)
        else:
            return np.random.choice(self.action_list)
    def learn(self,**args):
        pass
    def action_fn(self,state):
        pass
class Sarsa(RL_Module):
    def __init__(self, action_list, lr=1e-2, gamma=0.99, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=200):
        super(Sarsa, self).__init__(action_list, lr=lr, epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                                         epsilon_decay=epsilon_decay)
        self.q_table = pd.DataFrame(columns=self.action_list, dtype=np.float64)
        self.gamma = gamma
    def add_state(self, state):
        if not state in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.action_list), index=self.q_table.columns, name=state))

    def action_fn(self, state):
        return self.q_table.loc[state, :].argmax()

    def learn(self, state, action, reward, next_state, next_action, is_done=False):
        f"""
        Parameters:
           state: 当前状态
           action: 当前状态产生的行为
           reward: 这个行为的得到的奖励
           next_state: 下一个状态
           next_action: 下一个状态的动作
           is_done: 训练是否结束
        Return:
           无返回
           """
        self.add_state(next_state)
        q_target_value = self.q_table.loc[state, action]
        if next_state == 'terminal' or is_done:
            if state == 14 and reward == 1:
                print(f"action:{action} reward:{reward}")
            q_target_value += reward
        else:
            next_state_q_target_value = self.q_table.loc[next_state, next_action]
            pred = reward + self.gamma * next_state_q_target_value
            now = q_target_value
            q_target_value += self.lr * (pred - now)
        self.q_table.loc[state, action] = q_target_value
class Q_Learning(RL_Module):
    def __init__(self,action_list,lr=1e-2,gamma=0.99,epsilon_start=0.9,epsilon_end=0.1,epsilon_decay=200):
        super(Q_Learning,self).__init__(action_list,lr=lr,epsilon_start=epsilon_start,epsilon_end=epsilon_end,epsilon_decay=epsilon_decay)
        self.q_table = pd.DataFrame(columns=self.action_list,dtype=np.float64)
        self.gamma = gamma
    def add_state(self,state):
        if not state in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0]*len(self.action_list),index=self.q_table.columns,name=state))
    def action_fn(self,state):
        return self.q_table.loc[state,:].argmax()
    def learn(self,state,action,reward,next_state,is_done=False):
        f"""
        Parameters:
           state: 当前状态
           action: 当前状态产生的行为
           reward: 这个行为的得到的奖励
           next_state: 下一个状态
           is_done: 训练是否结束
        Return:
            无返回
        Q-Learning算法: 
        Q(s,a) <- Q(s,a) + a(r+ymax_a'Q(s',a')-Q(s,a))
        a:学习率
        y:衰减率
        
        
        """
        self.add_state(next_state)
        q_target_value = self.q_table.loc[state, action]
        if next_state == 'terminal' or is_done:
            if state == 14 and reward == 1:
                print(f"action:{action} reward:{reward}")
            q_target_value += reward
        else:
            next_state_q_target_value = self.q_table.loc[next_state, :].max()
            pred = reward + self.gamma*next_state_q_target_value
            now = q_target_value
            q_target_value += self.lr*(pred-now)
        self.q_table.loc[state, action] = q_target_value
class Q_Learning(RL_Module):
    def __init__(self,action_list,lr=1e-2,gamma=0.99,epsilon_start=0.9,epsilon_end=0.1,epsilon_decay=200):
        super(Q_Learning,self).__init__(action_list,lr=lr,epsilon_start=epsilon_start,epsilon_end=epsilon_end,epsilon_decay=epsilon_decay)
        self.q_table = pd.DataFrame(columns=self.action_list,dtype=np.float64)
        self.gamma = gamma
