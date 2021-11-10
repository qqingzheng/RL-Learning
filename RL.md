# 强化学习介绍

## Model-Free 与 Model-Based

Model-Free即不理解环境的学习，在真实环境中学习。**在强化学习中一个模型即一个环境**。
+ Q-Learning
+ Sarsa
+ Policy Gradients
+ 
Model-Based即先建立虚拟环境（模型）模拟真实环境。Model-Based具有想象力，可以根据反馈预知接下来发生的所有情况，并根据预测来采取下一步的策略。
+ Q-Learning
+ Sarsa
+ Policy Gradient

## Policy-Based 与 Value-Based

Policy-Based是对下一次各种行动进行分析得出采取不同动作的概率，然后根据概率做出行动。这样每种动作都可能被选中但可能性不同。
+ Policy Gradient

Value-Based则是对下一次各种行动实行评分，然后采用最高分的动作。
+ Q-Learning
+ Sarsa

## Monte-Carlo update 与 Temporal-Difference update

Monte-Carlo update是回合更新制。
+ Monte-carlo learning
+ 基础版的policy gradients

Temporal-Difference update是单步更新制。
+ Q-Learning
+ Saras
+ 升级版的policy gradients

## On-Policy 与 Off-Policy

On-Policy为在线学习，必须本人在场帮助计算机学习。
+ Sarsa

Off-Policy离线学习，不需要本人在场。
+ Q-Learning
+ Deep-Q-Network

