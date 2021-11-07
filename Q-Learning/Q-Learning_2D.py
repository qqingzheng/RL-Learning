import pandas as pd
import numpy as np
import time


"""
改编思路来自

莫烦
https://mofanpy.com/
"""

GAMMA = 0.1
ALPHA = 0.9
ACTIONS = ["UP", "DOWN", "RIGHT", "LEFT"]
MAX_EPISODES = 30
INTERVAL = 0.01
EPSILON = 0.9
END = [3, 3]


def build_q_table():
    table = np.zeros((4, 4, 4))
    return table
def get_reward(state, action):
    next_state = state.copy()
    if action == 'UP' and state[0]-1 >= 0:
        next_state[0] -= 1
    if action == 'DOWN' and state[0]+1 <= 3:
        next_state[0] += 1
    if action == 'RIGHT' and state[1]+1 <= 3:
        next_state[1] += 1
    if action == 'LEFT' and state[1]-1 >= 0:
        next_state[1] -= 1
    if next_state == END:
        r = 1
        return next_state,r
    else:
        r = 0
        return next_state,r

def take_action(state,table):
    if table[state[0],state[1],:].sum() == 0 or np.random.uniform(0, 1) >= EPSILON:
        return np.random.choice(ACTIONS)
    else:
        return ACTIONS[table[state[0],state[1],:].argmax(0)]
def update(table,state):
    map = np.zeros((table.shape[0],table.shape[1]))
    map[END[0],END[1]] = 2
    map[state[0],state[1]] = 1
    time.sleep(INTERVAL)
def rl():
    table = build_q_table()
    for episode in range(MAX_EPISODES):
        state = [0, 0]
        step = 0
        print(f" --- Episode {episode} --- ")
        while state != END:
            step += 1
            action = take_action(state,table)
            next_state,r = get_reward(state,action)
            table[state[0],state[1],ACTIONS.index(action)] += ALPHA*(r+np.max(table[next_state[0],next_state[1],:])-table[state[0],state[1],ACTIONS.index(action)])
            state = next_state
            update(table,state)
        print(f"Step: {step}")
    return table
if __name__ == "__main__":
    rl()