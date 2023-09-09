## 第六章 DQN 基本概念

DQN创新点：
1.  经验回放解决了两个问题：
    1. 序列决策的样本关联 
    2. 样本利用率低；
2. 固定Q目标解决算法非平稳性的问题。

经验回放充分利用了off-policy的优势，DQN实现了含有$\epsilon-greedy$的Sample函数保证所有动作能被探索到，实现了Learning方法使智能体与环境交互的数据能够交付给模型。

智能体与环境交互得到的数据存入Replay Buffer，从经验池中Sample一个batch的数据送给learn函数。Q网络用于产生Q预测值，target Q用于产生Q目标值，定期从Q网络复制参数至target Q网络，Q预测值与Q目标值作为Loss，根据Loss再去更新Q网络。
