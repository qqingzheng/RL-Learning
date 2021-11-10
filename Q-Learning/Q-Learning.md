# Q-Learning介绍

## 公式

$
Q(s,a) = Q(s,a) + \alpha(r+\gamma max_{a'}Q(s',a')-Q(s,a))
$

其中Q(s,a)表示当前状态下某个行动的得分，r为得分，$max_{a'}Q(s',a')$为下一个状态行动得分的最大值。$\alpha$为学习率，$\gamma$为对未来得分的衰减值。


#Deep-Q-Learning介绍

##回忆重演

当机器人(agent)的状态发生改变时，DQN会将这次状态变化进行储存，便于我们之后重新使用这个数据。
然后通过对储存的数据随机抽样(sampling it randomly)，这些样本组成的一个批次(batch)会去相关化(decorrelated)。

要完成上述步骤，我们需要两个类：
+ `Transition` a named tuple representing a single transition in our environment. It essentially maps (state, action) pairs to their (next_state, reward) result, with the state being the screen difference image as described later on.
+ `ReplayMemory`  a cyclic buffer of bounded size that holds the transitions observed recently. It also implements a `.sample()` method for selecting a random batch of transitions for training.

```python
from collections import namedtuple, deque
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```

## 公式
### 时差误差算法

### Huber损失函数
