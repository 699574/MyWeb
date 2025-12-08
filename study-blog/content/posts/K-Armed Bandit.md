---
title: 'K-Armed Bandit'
date: '2025-07-19'
tags: ['强化学习', 'K臂赌博机']
---

### $epision$-贪心算法

```python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class Bandit:
    def __init__(self,k):
        self.k = k
        self.rewards = np.random.rand(k)
        self.best_action = np.argmax(self.rewards)
        print(f"摇臂的真实获奖概率为: {[round(p, 3) for p in self.rewards]}")
        print(f"最佳摇臂是第 {self.best_action} 个，其概率为 {round(self.rewards[self.best_action], 3)}")

    def pull(self,arm_index):
        if arm_index < 0 or arm_index >= self.k:
            raise ValueError("摇臂编号超出范围！")

        if np.random.rand() < self.rewards[arm_index]:
            return 1
        else:
            return 0

class RL:
    def __init__(self,bandit,epsilon=0.1):
        self.bandit = bandit
        self.epsilon = epsilon
        self.k = self.bandit.k

        self.counts = np.zeros(self.bandit.k)
        self.q = np.zeros(self.bandit.k, dtype=float)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.bandit.k)
        else:
            return np.argmax(self.q)

    def update(self,arm_index,reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        old_q = self.q[arm_index]
        new_q = old_q + (1/n) * (reward - old_q)
        self.q[arm_index] = new_q

    def run(self,num_episodes):
        rewards = []
        for episode in range(num_episodes):
            arm = self.choose_action()
            reward = self.bandit.pull(arm)
            self.update(arm, reward)
            rewards.append(reward)
        return rewards
```

### 固定步长

```python
class RL2:
    def __init__(self,bandit,step,epsilon=0.1):
        self.bandit = bandit
        self.epsilon = epsilon
        self.k = self.bandit.k
        self.step = step

        self.counts = np.zeros(self.bandit.k)
        self.q = np.zeros(self.bandit.k, dtype=float)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.bandit.k)
        else:
            return np.argmax(self.q)

    def update(self,arm_index,reward):
        self.counts[arm_index] += 1
        old_q = self.q[arm_index]
        new_q = old_q + self.step * (reward - old_q)
        self.q[arm_index] = new_q

    def run(self,num_episodes):
        rewards = []
        for episode in range(num_episodes):
            arm = self.choose_action()
            reward = self.bandit.pull(arm)
            self.update(arm, reward)
            rewards.append(reward)
        return rewards
```

### 乐观初始值

```python
class RL3:
    def __init__(self,bandit,step,epsilon=0.1):
        self.bandit = bandit
        self.epsilon = epsilon
        self.k = self.bandit.k
        self.step = step

        self.counts = np.zeros(self.k)
        self.q = np.array([5]*self.k, dtype=float)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q)

    def update(self,arm_index,reward):
        self.counts[arm_index] += 1
        old_q = self.q[arm_index]
        new_q = old_q + self.step * (reward - old_q)
        self.q[arm_index] = new_q

    def run(self,num_episodes):
        rewards = []
        for episode in range(num_episodes):
            arm = self.choose_action()
            reward = self.bandit.pull(arm)
            self.update(arm, reward)
            rewards.append(reward)
        return rewards
```

### 置信度上界

```python
class UCB:
    def __init__(self,bandit,c,step):
        self.bandit = bandit
        self.c = c
        self.k = self.bandit.k
        self.step = step

        self.counts = np.zeros(self.k)
        self.q = np.zeros(self.bandit.k, dtype=float)
        self.total_steps = 0

    def choose_action(self):
        self.total_steps += 1

        ucb_values = np.zeros(self.k)
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

            exploitation_term = self.q[arm]
            exploration_term = self.c * np.sqrt(np.log(self.total_steps) / self.counts[arm])
            ucb_values[arm] = exploitation_term + exploration_term

        return np.argmax(ucb_values)

    def update(self,arm_index,reward):
        self.counts[arm_index] += 1
        old_q = self.q[arm_index]
        new_q = old_q + self.step * (reward - old_q)
        self.q[arm_index] = new_q

    def run(self,num_episodes):
        rewards = []
        for episode in range(num_episodes):
            arm = self.choose_action()
            reward = self.bandit.pull(arm)
            self.update(arm, reward)
            rewards.append(reward)
        return rewards
```

### 梯度赌博机

```python
class GradientBandit:
    def __init__(self, bandit, alpha):
        self.bandit = bandit
        self.k = bandit.k
        self.alpha = alpha
        self.preferences = np.zeros(self.k, dtype=float)
        self.avg_reward = 0
        self.total_steps = 0

    def choose_action(self):
        exp_prefs = np.exp(self.preferences)
        self.probabilities = exp_prefs / np.sum(exp_prefs)

        action = np.random.choice(self.k, p=self.probabilities)
        return action

    def update(self, arm_index, reward):
        self.total_steps += 1
        self.avg_reward += (1.0 / self.total_steps) * (reward - self.avg_reward)

        one_hot = np.zeros(self.k)
        one_hot[arm_index] = 1

        baseline_term = reward - self.avg_reward

        # 向量化的更新，同时更新所有偏好
        # 对选中动作 a:  (1 - π(a))
        # 对未选中动作 b: (0 - π(b)) = -π(b)
        update_term = self.alpha * baseline_term * (one_hot - self.probabilities)

        self.preferences += update_term

    def run(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            arm = self.choose_action()
            reward = self.bandit.pull(arm)
            self.update(arm, reward)
            rewards.append(reward)
        return rewards
```

### 实验

```python
if __name__ == "__main__":
    k = 10
    epsilon = 0.1
    episodes = 10000
    alpha = 0.1
    c = 2

    bandit = Bandit(k)
    solver = RL(bandit,epsilon)
    solver2 = RL2(bandit,alpha,epsilon)
    solver3 = RL3(bandit,alpha,epsilon)
    solver4 = UCB(bandit,c,alpha)
    solver5 = GradientBandit(bandit,alpha)

    rewards = solver.run(episodes)
    rewards2 = solver2.run(episodes)
    rewards3 = solver3.run(episodes)
    rewards4 = solver4.run(episodes)
    rewards5 = solver5.run(episodes)

    print("\n模拟结束。")
    print(f"智能体对各摇臂奖励的最终估计值: {[round(q, 3) for q in solver.q]}")
    print(f"各摇臂被选择的次数: {solver.counts}")

    total_reward = sum(rewards)
    average_reward = total_reward / episodes
    print(f"\n在 {episodes} 次尝试中，总奖励为: {total_reward}")
    print(f"平均每次奖励为: {average_reward:.4f}")

    # 理论上的最优平均奖励
    best_possible_avg_reward = bandit.rewards[bandit.best_action]
    print(f"理论上的最佳平均奖励: {best_possible_avg_reward:.4f}")

    # --- 5. 可视化结果 ---
    # 计算每次尝试后的累计平均奖励
    cumulative_average_reward = np.cumsum(rewards) / (np.arange(episodes) + 1)
    cumulative_average_reward2 = np.cumsum(rewards2) / (np.arange(episodes) + 1)
    cumulative_average_reward3 = np.cumsum(rewards3) / (np.arange(episodes) + 1)
    cumulative_average_reward4 = np.cumsum(rewards4) / (np.arange(episodes) + 1)
    cumulative_average_reward5 = np.cumsum(rewards5) / (np.arange(episodes) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_average_reward, label=f'ε-贪心算法 (ε={epsilon})')
    plt.plot(cumulative_average_reward2, label=f'ε-贪心算法2 (ε={epsilon})')
    plt.plot(cumulative_average_reward3,label=f'ε-贪心算法3 (ε={epsilon})')
    plt.plot(cumulative_average_reward4,label=f'UCB (c={c})')
    plt.plot(cumulative_average_reward5,label=f'Gradient')
    plt.axhline(y=best_possible_avg_reward, color='r', linestyle='--', label='理论最佳平均奖励')
    plt.title('K臂赌博机: ε-贪心算法性能')
    plt.xlabel('尝试次数 (Trials)')
    plt.ylabel('平均奖励 (Average Reward)')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 非平稳

```python
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class Bandit:
    def __init__(self, k, drift_std=0.01):
        self.k = k
        self.drift_std = drift_std
        self.true_q_values = np.zeros(self.k)

    def pull(self,arm_index):
        reward = np.random.randn() + self.true_q_values[arm_index]
        return reward

    def random_walk(self):
        drift = np.random.normal(0, self.drift_std, self.k)
        self.true_q_values += drift

    def get_optimal_action(self):
        return np.argmax(self.true_q_values)

class Solver:
    def __init__(self, k, epsilon, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha

        self.q = np.zeros(self.k)
        self.counts = np.zeros(self.k, dtype=int)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q)

    def update(self, arm_index, reward):
        self.counts[arm_index] += 1

        if self.alpha is None:
            step_size = 1.0 / self.counts[arm_index]
        else:
            step_size = self.alpha

        old_q = self.q[arm_index]
        self.q[arm_index] = old_q + step_size * (reward - old_q)

def run_experiment(k, steps, runs, epsilon, alpha):

    rewards = np.zeros((runs, steps))
    actions_avg = np.zeros((runs, steps))

    rewards_alpha = np.zeros((runs, steps))
    actions_alpha = np.zeros((runs, steps))

    # 使用tqdm来显示外层循环的进度
    for run in tqdm(range(runs), desc="正在运行实验"):
        # 为每次运行创建新的环境和智能体
        bandit = Bandit(k)

        solver_avg = Solver(k, epsilon, alpha=None)

        solver_alpha = Solver(k, epsilon, alpha=alpha)

        for step in range(steps):
            optimal_action = bandit.get_optimal_action()

            action1 = solver_avg.choose_action()
            reward1 = bandit.pull(action1)
            solver_avg.update(action1, reward1)
            rewards[run, step] = reward1
            if action1 == optimal_action:
                actions_avg[run, step] = 1

            action2 = solver_alpha.choose_action()
            reward2 = bandit.pull(action2)
            solver_alpha.update(action2, reward2)
            rewards_alpha[run, step] = reward2
            if action2 == optimal_action:
                actions_alpha[run, step] = 1

            bandit.random_walk()

    avg_sample = np.mean(rewards, axis=0)
    avg_constant = np.mean(rewards_alpha, axis=0)

    percent_sample = np.mean(actions_avg, axis=0) * 100
    percent_constant = np.mean(actions_alpha, axis=0) * 100

    return (avg_sample, percent_sample), (avg_constant, percent_constant)

def plot_results(results_sample, results_constant, steps):
    avg_sample, percent_sample = results_sample
    avg_constant, percent_constant = results_constant

    plt.figure(figsize=(14, 12))

    plt.subplot(2, 1, 1)
    plt.plot(avg_sample, label='采样平均法 (步长 = 1/n)')
    plt.plot(avg_constant, label='常数步长法 (α = 0.1)')
    plt.xlabel('步数 (Steps)')
    plt.ylabel('平均奖励 (Average Reward)')
    plt.title('非平稳环境下不同步长策略的平均奖励对比')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(percent_sample, label='采样平均法 (步长 = 1/n)')
    plt.plot(percent_constant, label='常数步长法 (α = 0.1)')
    plt.xlabel('步数 (Steps)')
    plt.ylabel('最优动作选择率 (%)')
    plt.title('非平稳环境下不同步长策略的最优动作选择率')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 实验参数
    K = 10
    STEPS = 10000
    RUNS = 200
    EPSILON = 0.1
    ALPHA = 0.1

    # 运行实验
    results_sample, results_constant = run_experiment(K, STEPS, RUNS, EPSILON, ALPHA)

    # 绘制结果
    plot_results(results_sample, results_constant, STEPS)
```