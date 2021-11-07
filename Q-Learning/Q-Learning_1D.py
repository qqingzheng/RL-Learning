import pandas as pd
import numpy as np
import time

from pandas.core.frame import DataFrame
"""
本例子来自于莫烦
https://mofanpy.com/
"""
N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 13   # 最大回合数
FRESH_TIME = 0.01    # 移动间隔时间
def build_q_table(states,actions) -> pd.DataFrame:
    table = pd.DataFrame(
        np.zeros((states,len(actions))),
        columns=ACTIONS
    )
    return table
def choose_action(state,table):
    value_of_actions = table.iloc[state,:]
    if (np.random.uniform() > EPSILON) or (value_of_actions.all() == 0):
        return np.random.choice(ACTIONS)
    else:
        return value_of_actions.idxmax()
def get_env_feedback(S, A):
    """
    S_为下一个状态
    A为本状态得分
    """
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R
def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    print(q_table)
    """
    初始化Q_table，得到：
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0

Q_table是在不同状态的时候的不同行为得分
    """
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0
        S = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 环境更新
        while not is_terminated:
            A = choose_action(S, q_table)   # 选行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A]    # 估算的(状态-行为)值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  
            update_env(S, episode, step_counter+1)  # 环境更新
            step_counter += 1
    return q_table
if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)