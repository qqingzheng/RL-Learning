import Core.Mpl as mp
import gym
import numpy as np
import Core.Module as m
import time
EPISODE = 50000
env = gym.make("Taxi-v3").unwrapped
action_list = tuple(range(env.action_space.n))
qmodule = m.Q_Learning(action_list=action_list,lr=1e-2,epsilon_decay=2000,epsilon_end=0.2)
print("Training")
vt = mp.ViewTrend(2,"Training","EPISODE","REWARD")
vt1 = mp.ViewTrend(1,"Suc","EPISODE","REWARD")
episode_log = []
reward_log = [[],[],[]]
train_reward_log = [0,0,0]
for episode in range(EPISODE):
    observation = env.reset()
    episode_reward = 0
    while True:
        action = qmodule.choose_action(observation)
        next_observation, reward, done,_ = env.step(action)
        episode_reward += reward
        qmodule.learn(observation,action,reward,next_observation,done)
        observation = next_observation
        if done:
            break
    if episode_reward > 0:
        train_reward_log[0] += 1
    elif episode_reward > -200:
        train_reward_log[1] += 1
    elif episode_reward > -1000:
        train_reward_log[2] += 1
    episode_log.append(episode_reward)
    reward_log[0].append(train_reward_log[0])
    reward_log[1].append(train_reward_log[1])
    reward_log[2].append(train_reward_log[2])
    print(f"\r Training({episode}): {episode_reward}",end="")
    if episode % 1 == 0:
        vt.update(episode_log)
        vt1.update(reward_log[0], reward_log[1], reward_log[2])
print("\n Done!")
def test_module():
    total_reward = 0
    for episode in range(500):
        episode_reward = 0
        observation = env.reset()
        while True:
            action = qmodule.choose_action(observation,is_test=True)
            next_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            observation = next_observation
            if done or episode_reward < -1000:
                break
        total_reward += episode_reward
        print(f"\r Testing({episode}): {episode_reward}", end="")
    print(f"\nTesting reward average: {total_reward/100}")
print("Testing")
test_module()