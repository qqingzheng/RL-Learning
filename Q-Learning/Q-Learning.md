# Q-Learning介绍

## 公式

$
Q(s,a) = Q(s,a) + \alpha(r+\gamma max_{a'}Q(s',a')-Q(s,a))
$
其中Q(s,a)表示当前状态下某个行动的得分，r为得分，$max_{a'}Q(s',a')$为下一个状态行动得分的最大值。$\alpha$为学习率，$\gamma$为对未来得分的衰减值。