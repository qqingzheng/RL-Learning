# Gym库的使用
来源于：https://blog.csdn.net/weixin_44791964/article/details/96767972

**环境初始化**
```python
env = gym.make('CartPole-v0')   
# 定义使用gym库中的某一个环境，'CartPole-v0'可以改为其它环境
env = env.unwrapped             
# 据说不做这个动作会有很多限制，unwrapped是打开限制的意思
```

**环境的各个参数**
```python
env.action_space   		
# 查看这个环境中可用的action有多少个，返回Discrete()格式
env.observation_space   
# 查看这个环境中observation的特征，返回Box()格式
n_actions=env.action_space.n 
# 查看这个环境中可用的action有多少个，返回int
n_features=env.observation_space.shape[0] 
# 查看这个环境中observation的特征有多少个，返回int
```
**环境刷新**
```python
env.reset()
# 用于一个世代（done）后环境的重启，获取回合的第一个observation
env.render()	
# 用于每一步后刷新环境状态
```
**环境转换**
```python
observation_, reward, done, info = env.step(action)
# 获取下一步的环境、得分、检测是否完成。
```