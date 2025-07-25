---
title: '策略梯度'
date: '2025-07-13'
tags: ['强化学习', '策略梯度']
---

# Lecture 5

## 1. 关键要点 (Key Points)

-   **策略梯度范式**: 直接参数化策略 $\pi_\theta(a|s)$，并通过优化参数 $\theta$ 来最大化预期回报。
-   **为什么需要策略梯度？**:
    1.  **随机策略（Stochastic Policies）的优势**: 在某些环境下，最优策略本身就是随机的（例如：石头剪刀布、部分可观察MDPs中的混淆状态）。
    2.  **连续行动空间（Continuous Action Spaces）**: 基于价值的方法通常难以处理连续行动空间，而策略梯度可以自然地通过对行动分布进行参数化来解决。
    3.  **内生随机性**: 有助于探索。
-   **策略梯度目标**: 找到策略参数 $\theta$ 使得期望回报（如起始状态价值 $V(s_0; \theta)$）最大化。这是一个优化问题。
-   **似然比策略梯度定理（Likelihood Ratio Policy Gradient Theorem）**: 核心数学推导，将期望回报的梯度转换为期望中包含策略对数导数（即分数函数）的形式。
    $$\nabla_\theta V(\theta) = E_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t|s_t) G_t]$$
    （这里用 $G_t$ 指代从当前时间步 $t$ 往后的总回报，也可以是整个回合的总回报 $R(\tau)$）。
-   **分数函数（Score Function）**: $\nabla_\theta \log \pi_\theta(a|s)$，表示调整参数如何影响选择该行动的对数概率。它的计算不需要环境模型。
-   **REINFORCE算法**: 最基本的蒙特卡洛策略梯度算法。它是一个**无偏（Unbiased）**的梯度估计器，但通常**方差很高（High Variance）**。
-   **方差削减（Variance Reduction）**: 策略梯度算法方差高是主要挑战。常用技术包括：
    1.  **时序结构**: 利用回报的因果关系，只用未来回报 $G_t$ 乘以当前行动的梯度，而不是整个回合的回报 $R(\tau)$。
    2.  **基线（Baseline）**: 从回报中减去一个基线 $b(s)$，即用 $G_t - b(s)$ 作为权重。这可以显著降低方差，而**不引入偏差**。最佳基线是状态价值函数 $V^\pi(s)$。
    3.  **替代蒙特卡洛回报**: 使用 $n$-步回报或Q函数作为目标，用以权衡偏差和方差。
-   **优势函数（Advantage Function）**: 当基线为 $V^\pi(s)$ 时，策略梯度可以表示为 $E_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t|s_t) A^\pi(s_t, a_t)]$，其中优势函数 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$。
-   **Actor-Critic方法**: 一类结合了策略（Actor）和价值函数（Critic）学习的算法。Critic估计价值函数（用作基线），Actor基于Critic的反馈更新策略。

---

## 2. 详细解读 (Detailed Breakdown)

### 2.1 策略梯度的引入与动机 (Introduction and Motivation for Policy Gradient)

*   **回顾**: 之前的讲座聚焦于基于价值（Value-Based）的强化学习，通过学习价值函数（$V(s)$ 或 $Q(s,a)$）并从中推导出策略（例如，贪婪策略）。
*   **策略梯度的不同**: 策略梯度方法**直接参数化策略 $\pi_\theta(a|s)$**（如神经网络），并通过优化策略参数 $\theta$ 来最大化期望回报。
*   **为什么需要策略梯度？**:
    *   **随机策略的优势**: 在许多情况下，最优策略是**随机的**。例如，在石头剪刀布游戏中，任何确定性策略都会被对手利用；在部分可观察MDPs（POMDPs）中，智能体可能无法区分两个状态（**混淆状态/aliasing**），此时随机策略可以避免陷入局部最优（例如，在混淆的网格世界中，随机向左或向右移动比固定方向更优）。
    *   **连续行动空间**: 基于价值的方法通常通过离散化来处理连续行动空间，但这可能效率低下或引入近似误差。策略梯度可以自然地通过输出一个连续分布（如高斯分布的均值和方差）来处理连续行动。
    *   **平滑的策略空间**: 策略梯度通常能提供更平滑的优化过程。
*   **应用**: 策略梯度方法在自然语言处理（NLP）中的序列训练、机器人控制（如AIBO机器狗学走路）、以及ChatGPT中使用的PPO算法等领域都产生了重要影响。

### 2.2 策略优化问题 (Policy Optimization Problem)

*   **目标**: 给定一个参数化的策略 $\pi_\theta(s,a)$，目标是找到最佳参数 $\theta$ 来最大化策略的价值函数，通常是**起始状态价值 $V(s_0; \theta)$**。这是一个优化问题。
*   **优化方法**:
    *   **无梯度优化 (Gradient-Free Optimization)**: 如爬山算法（Hill Climbing）、单纯形法（Simplex）、遗传算法（Genetic Algorithms）、交叉熵方法（CEM）和协方差矩阵自适应进化策略（CMA-ES）。这些方法通常易于并行化，可以处理不可微分的策略，但样本效率较低。
    *   **梯度优化 (Gradient-Based Optimization)**: 利用目标函数的梯度信息进行优化，通常更高效。策略梯度方法属于这一类。

### 2.3 似然比策略梯度 (Likelihood Ratio Policy Gradient)

策略梯度算法的核心是计算**期望回报对策略参数的梯度 $\nabla_\theta V(\theta)$**。

1.  **定义目标函数**: 期望回报 $V(\theta)$ 可以表示为所有可能轨迹的回报与其发生概率的乘积和：
    $$V(\theta) = E_{\pi_\theta}[R(\tau)] = \sum_{\tau} P(\tau; \theta) R(\tau)$$
    其中 $\tau$ 表示一个完整的状态-行动轨迹，$P(\tau; \theta)$ 是在策略 $\pi_\theta$ 下轨迹 $\tau$ 发生的概率，$R(\tau)$ 是轨迹 $\tau$ 的总回报。

2.  **求梯度**: 对目标函数求关于参数 $\theta$ 的梯度：
    $$\nabla_\theta V(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) R(\tau) = \sum_{\tau} \nabla_\theta P(\tau; \theta) R(\tau)$$

3.  **似然比技巧（Likelihood Ratio Trick）**: 这是策略梯度的关键一步。它利用了恒等式 $\nabla_x f(x) = f(x) \nabla_x \log f(x)$：
    $$\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)$$

4.  **代入并重写为期望**: 将上述结果代回梯度公式：
    $$\nabla_\theta V(\theta) = \sum_{\tau} P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau)$$
    这可以被重新解释为期望的形式：
    $$\nabla_\theta V(\theta) = E_{\pi_\theta}[R(\tau) \nabla_\theta \log P(\tau; \theta)]$$
    这个公式的强大之处在于，我们可以通过**采样轨迹**来估计这个期望，而不需要知道 $P(\tau; \theta)$ 的完整分布。

5.  **轨迹概率的分解**: 对于马尔可夫决策过程（MDP），轨迹 $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_T)$ 的概率 $P(\tau; \theta)$ 可以分解为：
    $$P(\tau; \theta) = \mu(S_0) \prod_{t=0}^{T-1} \pi_\theta(A_t|S_t) P(S_{t+1}|S_t, A_t)$$
    取对数并求梯度：
    $$\nabla_\theta \log P(\tau; \theta) = \nabla_\theta \log \mu(S_0) + \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t|S_t) + \sum_{t=0}^{T-1} \nabla_\theta \log P(S_{t+1}|S_t, A_t)$$
    由于 $\mu(S_0)$ 和 $P(S_{t+1}|S_t, A_t)$ 不依赖于策略参数 $\theta$，所以它们的梯度为零。
    因此，
    $$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t|S_t)$$
    这表明，我们只需要计算策略本身的对数概率梯度，**不需要知道环境的动态模型！** 这使得策略梯度成为一种**无模型**的方法。

6.  **分数函数（Score Function）**: $\nabla_\theta \log \pi_\theta(a|s)$ 被称为分数函数。它衡量了对策略参数的微小改变如何影响选择特定行动的对数概率。
    *   **Softmax策略的例子**: 对于离散行动空间，如果策略是Softmax函数，参数化为 $\pi_\theta(s,a) = \exp(\phi(s,a)^T \theta) / \sum_{a'} \exp(\phi(s,a')^T \theta)$，则其分数函数为 $\nabla_\theta \log \pi_\theta(s,a) = \phi(s,a) - E_{\pi_\theta}[\phi(s,\cdot)]$。
    *   **高斯策略的例子**: 对于连续行动空间，如果策略是高斯分布，其均值由参数 $\theta$ 决定，则其分数函数为 $\nabla_\theta \log \pi_\theta(s,a) = (a - \mu(s))\phi(s)/\sigma^2$。

7.  **最终梯度估计器**: 结合采样和分数函数，策略梯度的估计器为：
    $$\hat{g} = \frac{1}{m} \sum_{i=1}^{m} R(\tau^{(i)}) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t^{(i)}|S_t^{(i)})$$
    这可以看作是**REINFORCE算法**的基础。

### 2.4 REINFORCE 算法 (Monte-Carlo Policy Gradient)

REINFORCE是一种最基本的蒙特卡洛策略梯度算法，它使用完整的轨迹回报来估计梯度。

*   **算法步骤**:
    1.  **初始化**: 随机初始化策略参数 $\theta$。
    2.  **循环迭代**:
        *   **生成回合**: 遵循当前策略 $\pi_\theta$，生成一个完整的经验回合：$S_0, A_0, R_1, \dots, S_T$。
        *   **计算回报**: 对于回合中的每个时间步 $t$，计算从该时间步开始到回合结束的总回报 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}$。
        *   **更新策略参数**: 对于回合中的每个时间步 $t=0, \dots, T-1$：
            $$\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$$
            （注意，这里实际上是 $\Delta\theta = \alpha G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$，然后将 $\Delta\theta$ 加到 $\theta$ 上。）

*   **特性**: REINFORCE是一个**无偏的梯度估计器**。这意味着，其梯度的期望值等于真实梯度的值。然而，它的**方差很高**，导致学习过程不稳定且收敛缓慢，尤其是在长回合和高维度空间中。

### 2.5 方差削减技术 (Variance Reduction Techniques)

高方差是策略梯度方法的主要挑战。以下是几种常用的方差削减技术：

1.  **利用时序结构 (Temporal Structure)**:
    *   在REINFORCE中，梯度估计器使用整个轨迹的总回报 $R(\tau)$（或者从 $S_t$ 开始的 $G_t$）。然而，对于 $S_t, A_t$ 的梯度更新，只有在 $t$ 时刻之后获得的奖励才应该影响其价值。
    *   修正后的梯度估计器（已经体现在上面REINFORCE的 $G_t$ 中）利用了因果关系：只使用**未来回报 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}$** 来乘以当前时间步的策略梯度 $\nabla_\theta \log \pi_\theta(A_t|S_t)$。这比使用整个回合的总回报 $R(\tau)$ 更好地利用了时序结构，因为 $R_{t'}$，其中 $t'<t$ 的奖励与 $A_t$ 的选择无关。

2.  **引入基线 (Baseline)**:
    *   **原理**: 从回报中减去一个**基线 $b(S_t)$**，即用 $G_t - b(S_t)$ 来代替 $G_t$ 作为更新的权重：
        $$\theta \leftarrow \theta + \alpha (G_t - b(S_t)) \nabla_\theta \log \pi_\theta(A_t|S_t)$$
    *   **效果**: 引入基线**不会引入偏差**，即 $E_{\pi_\theta}[(G_t - b(S_t)) \nabla_\theta \log \pi_\theta(A_t|S_t)] = E_{\pi_\theta}[G_t \nabla_\theta \log \pi_\theta(A_t|S_t)]$，但可以显著**降低方差**。这是因为 $\sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0$。如果基线选择得当，可以有效抵消梯度中的噪声。
    *   **最佳基线**: 理论上，使方差最小的基线是**状态价值函数 $V^\pi(S_t)$**。
    *   **实践中的基线**: 在实践中，通常使用一个参数化的价值函数逼近器 $\hat{V}(S_t; w)$ 来估计 $V^\pi(S_t)$ 作为基线。

3.  **替代蒙特卡洛回报 (Alternatives to MC Returns)**:
    *   REINFORCE使用完整的回合回报 $G_t$ 作为目标，这会带来高方差。
    *   可以像TD学习那样，引入**自举**来降低方差，但同时会引入一些偏差。
    *   **$N$-步回报**: 使用 $N$ 步的实际奖励加上第 $N$ 步状态的价值估计作为目标。例如 $R^{(1)}_t = R_{t+1} + \gamma V(S_{t+1})$。
    *   **Q函数**: 直接使用状态-行动价值函数 $Q^\pi(S_t, A_t)$ 作为目标。当基线是 $V^\pi(S_t)$ 时，这自然引出了**优势函数（Advantage Function）**。

### 2.6 优势函数与 Actor-Critic 方法 (Advantage Function and Actor-Critic Methods)

*   **优势函数**: 将基线设置为状态价值函数 $V^\pi(S_t)$ 时，策略梯度可以写成：
    $$\nabla_\theta V(\theta) = E_{\pi_\theta}[\nabla_\theta \log \pi_\theta(A_t|S_t) (Q^\pi(S_t, A_t) - V^\pi(S_t))]$$
    其中 $A^\pi(S_t, A_t) = Q^\pi(S_t, A_t) - V^\pi(S_t)$ 被称为**优势函数**。它衡量了在状态 $S_t$ 下采取行动 $A_t$ 比平均而言更好的程度。
    使用优势函数作为权重可以显著降低方差。

*   **Actor-Critic 方法**:
    *   **Actor (策略网络)**: 负责学习和更新策略 $\pi_\theta$，它就像一个“行动者”，在环境中采取行动。
    *   **Critic (价值网络)**: 负责学习和估计价值函数 $V^\pi(s)$ 或 $Q^\pi(s,a)$，它就像一个“评论家”，评估Actor行动的好坏，并提供优势函数估计作为Actor的梯度更新信号。
    *   **学习过程**: Critic通过TD学习等方法估计价值函数，而Actor则利用Critic提供的优势函数估计来更新其策略参数。
    *   **优点**: 结合了基于价值和基于策略方法的优点，通常比REINFORCE具有更低的方差和更好的收敛性，并且能够处理连续行动空间。
    *   **示例**: A3C (Asynchronous Advantage Actor-Critic) 是一个非常流行的Actor-Critic算法。

---

## 3. 总结与启示 (Conclusion & Implications)

-   **核心区别**: 策略梯度直接优化参数化策略，使其在处理随机策略和连续行动空间方面具有独特优势。
-   **数学基石**: 似然比技巧是推导策略梯度的关键，它使得我们能够从采样的经验中估计梯度，而无需环境模型。
-   **REINFORCE**: 作为最基础的策略梯度算法，虽然简单直观，但其高方差是主要限制。
-   **方差削减是关键**: 时序结构、基线（尤其是价值函数）以及TD式的目标替代是策略梯度算法成功的关键。
-   **Actor-Critic**: 提供了一个强大的框架，将策略学习和价值估计结合起来，以提高策略梯度方法的效率和稳定性。
-   **挑战**: 策略梯度方法通常收敛到局部最优，且策略评估的效率和方差仍然是研究的热点。

---

## Lecture 6

## 1. 关键要点 (Key Points)

-   **香草策略梯度的局限性**:
    1.  **样本效率低**: 每次梯度更新后，当前批次的经验数据就会被丢弃，因为策略的期望是**同策略（on-policy）**的。
    2.  **步长选择困难**: 策略参数的小幅变化可能导致策略的巨大变化，使得学习过程不稳定，容易出现“性能崩溃”。
    3.  **策略空间与参数空间的非线性关系**: 策略参数空间中的欧几里得距离，与策略行为（如KL散度）上的距离不一致。

-   **策略性能界限（Performance Bounds）**:
    *   定义了新策略 $\pi'$ 相对于旧策略 $\pi$ 的性能提升 $J(\pi') - J(\pi)$ 的精确表达式，利用旧策略的优势函数 $A^\pi(s,a)$。
    *   通过**重要性采样（Importance Sampling）**，可以将其转换为在旧策略 $\pi$ 下采样得到的期望。

-   **关键近似（Useful Approximation）**:
    *   引入了在旧策略下优化新策略的近似目标函数 $\mathcal{L}_\pi(\pi')$，它允许在旧策略下收集的数据进行多步梯度更新。
    *   这种近似在新旧策略**足够接近（通过KL散度衡量）**时效果良好。

-   **单调改进理论（Monotonic Improvement Theory）**:
    *   证明了通过最大化近似目标函数 $\mathcal{L}_\pi(\pi')$ 并施加KL散度约束，可以**近似地保证策略的单调改进**。
    *   即 $J(\pi') - J(\pi) \ge \mathcal{L}_\pi(\pi') - C \sqrt{E_{s \sim d^\pi}[D_{KL}(\pi'||\pi)[s]]}$。

-   **重要性采样在策略梯度中的挑战**:
    *   虽然重要性采样允许异策略学习，但当行为策略和目标策略差异较大时，**重要性采样权重可能爆炸或消失**，导致梯度估计方差巨大，甚至发散。

-   **近端策略优化（Proximal Policy Optimization, PPO）**: 一种旨在解决策略梯度方法稳定性和样本效率问题的算法家族。它通过**近似地约束策略更新的幅度**，来保证学习过程的稳定性。
    *   **两种主要变体**:
        1.  **自适应KL惩罚（Adaptive KL Penalty）**: 在目标函数中加入KL散度惩罚项，并根据KL散度的大小动态调整惩罚系数 $\beta_k$。
        2.  **截断目标函数（Clipped Objective）**: 这是PPO更常用的变体。通过截断策略比率（新旧策略概率比 $r_t(\theta) = \frac{\pi_\theta(A_t|S_t)}{\pi_{\theta_k}(A_t|S_t)}$），限制了策略更新的幅度，避免了过大的策略变化。

-   **PPO的成功**: PPO算法因其在经验上表现出色、实现相对简单且稳定性高而广受欢迎，是当前深度强化学习领域的基线算法之一，并被应用于ChatGPT等大型模型训练中。

---

## 2. 详细解读 (Detailed Breakdown)

### 2.1 策略梯度方法的局限性 (Problems with Policy Gradient Methods)

在“策略梯度 I”中，我们学习了REINFORCE算法，它是一个无偏的梯度估计器。然而，它存在一些显著的局限性：

1.  **样本效率低下 (Poor Sample Efficiency)**:
    *   策略梯度算法的期望是**同策略（on-policy）**的，即梯度估计依赖于当前策略产生的经验。
    *   这意味着每次进行一次参数更新后，之前收集的数据就变得“陈旧”而无法再直接使用，必须丢弃并重新从新策略中采集新数据。这导致大量经验被浪费。

2.  **步长选择困难 (Difficulty in Choosing Step Size)**:
    *   策略梯度算法使用随机梯度上升来更新策略参数：$\theta_{k+1} = \theta_k + \alpha_k \hat{g}_k$。
    *   一个过大的步长 $\alpha_k$ 可能导致**性能崩溃（performance collapse）**：策略参数的小幅变化可能导致策略的巨大变化，从而使智能体在环境中表现非常糟糕，甚至无法恢复。
    *   一个过小的步长 $\alpha_k$ 则会导致学习进度缓慢，难以收敛。
    *   “正确”的步长大小是动态变化的，取决于当前参数 $\theta$ 的位置，这使得手动调整或自适应学习率策略（如Adam）难以完美解决问题。

3.  **策略空间与参数空间的非线性关系 (Policy Space vs. Parameter Space)**:
    *   这是一个更深层次的问题。策略函数 $\pi_\theta(a|s)$ 通常通过复杂的非线性函数（如神经网络）参数化。
    *   在参数空间（$\theta$）中的小距离（欧几里得距离）并不总是对应于策略空间（$\pi_\theta$）中的小距离（如KL散度）。也就是说，参数的小改动可能导致策略行为的巨大改变。这进一步加剧了步长选择的难度和不稳定性。

### 2.2 策略性能界限 (Policy Performance Bounds)

为了解决样本效率和步长问题，我们需要一种方法来：
1.  **重用旧数据**: 在旧策略下收集的数据可以用于优化新策略。
2.  **约束策略变化**: 确保策略更新不会过大，从而保证学习的稳定性。

**性能差异引理（Performance Difference Lemma）**: 揭示了任意两个策略 $\pi'$ 和 $\pi$ 之间的价值差异：
$$J(\pi') - J(\pi) = E_{\tau \sim \pi'}\left[\sum_{t=0}^\infty \gamma^t A^\pi(S_t, A_t)\right]$$
其中 $J(\pi)$ 是策略 $\pi$ 的价值（通常是起始状态的预期回报），$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 是优势函数。
这个引理说明了新策略的性能提升可以通过在**新策略**下运行并计算相对于旧策略优势函数的折扣和来衡量。

通过一些数学转换（包括重要性采样和利用折扣未来状态分布 $d^\pi(s)$ 的定义），可以将上述公式改写为：
$$J(\pi') - J(\pi) = \frac{1}{1-\gamma} E_{s \sim d^{\pi'}, a \sim \pi'}\left[\frac{\pi'(a|s)}{\pi(a|s)} A^\pi(s,a)\right]$$
这表明了性能差异如何通过新策略下的采样，以及新旧策略概率比（重要性采样权重）和旧策略的优势函数来表示。

### 2.3 单调改进理论与KL散度 (Monotonic Improvement Theory and KL-Divergence)

上述公式仍然存在问题：它需要从**新策略 $\pi'$** 中采样，而这恰好是我们想要优化的对象。为了解决这个问题，并允许从**旧策略 $\pi$** 中采样，我们引入一个**近似**。

**关键近似**: 当新策略 $\pi'$ 与旧策略 $\pi$ **足够接近**时，我们可以近似地用旧策略的折扣状态分布 $d^\pi(s)$ 来代替新策略的折扣状态分布 $d^{\pi'}(s)$。
这导致了一个近似目标函数 $\mathcal{L}_\pi(\pi')$：
$$\mathcal{L}_\pi(\pi') = \frac{1}{1-\gamma} E_{s \sim d^\pi, a \sim \pi}\left[\frac{\pi'(a|s)}{\pi(a|s)} A^\pi(s,a)\right]$$
这个近似目标函数 $L_\pi(\pi')$ 使得我们可以**利用旧策略 $\pi$ 收集的数据来更新新策略 $\pi'$ 的参数**。这是解决样本效率低问题的关键。

**理论保证**: Schulman et al. (2015) 证明了一个策略性能的**下界**：
$$J(\pi') - J(\pi) \ge \mathcal{L}_\pi(\pi') - C \sqrt{E_{s \sim d^\pi}[D_{KL}(\pi'||\pi)[s]]}$$
其中 $D_{KL}(\pi'||\pi)[s] = \sum_a \pi'(a|s) \log \frac{\pi'(a|s)}{\pi(a|s)}$ 是状态 $s$ 下策略 $\pi'$ 相对于 $\pi$ 的KL散度，$C$ 是一个常数。
这个不等式非常重要：它表明，如果我们能够最大化近似目标函数 $\mathcal{L}_\pi(\pi')$，同时**限制新旧策略之间的KL散度 $D_{KL}(\pi'||\pi)[s]$ 不超过某个阈值**，我们就可以**近似地保证策略性能的单调改进**。

这个理论是**信任区域策略优化（Trust Region Policy Optimization, TRPO）**和**近端策略优化（PPO）**的基础。它们不再直接最大化 $J(\pi')$, 而是通过最大化这个下界，同时控制KL散度来确保更新的稳定性。

### 2.4 重要性采样在策略梯度中的挑战 (Challenges of Importance Sampling in Policy Gradient)

尽管重要性采样允许我们使用旧策略的数据来训练新策略，但它并非没有挑战。
重要性采样权重 $\frac{P(x)}{Q(x)}$ 如果 $Q(x)$（行为策略）在 $P(x)$（目标策略）有高概率但 $Q(x)$ 自身概率很低的地方，那么权重就可能**爆炸**，导致梯度估计的方差巨大，甚至使学习过程发散。即使新旧策略只是略有不同，这些小的差异也会在多个时间步上累积并相乘，导致巨大的重要性权重，从而造成不稳定性。

### 2.5 近端策略优化（PPO） (Proximal Policy Optimization (PPO))

PPO算法旨在解决香草策略梯度中样本效率低和步长选择困难的问题，同时避免重要性采样带来的不稳定性。PPO通过**近似地约束策略更新的幅度**，从而在保证策略改进的同时，避免大的策略变化。

PPO主要有两种变体：

1.  **自适应KL惩罚（Adaptive KL Penalty）**:
    *   **目标函数**:
        $$\theta_{k+1} = \arg \max_\theta \mathcal{L}_{\theta_k}(\theta) - \beta_k D_{KL}(\theta||\theta_k)$$
        其中 $\mathcal{L}_{\theta_k}(\theta)$ 是上述的近似目标函数，$D_{KL}(\theta||\theta_k)$ 是新策略 $\pi_\theta$ 相对于旧策略 $\pi_{\theta_k}$ 的KL散度。
    *   **动态调整惩罚系数 $\beta_k$**: 算法会根据实际计算出的KL散度大小来调整 $\beta_k$。
        *   如果KL散度太大（超出目标值 $\delta$ 的 $1.5$ 倍），则增加 $\beta_k$（例如 $\beta_k \leftarrow 2\beta_k$），以更严厉地惩罚大的策略变化。
        *   如果KL散度太小（低于目标值 $\delta$ 的 $0.5$ 倍），则减小 $\beta_k$（例如 $\beta_k \leftarrow \beta_k/2$），允许更大的策略变化。
    *   **优点**: 自动调整惩罚强度，学习过程更稳定。
    *   **缺点**: KL散度的计算和 $\beta_k$ 的动态调整会增加一定的复杂性。

2.  **截断目标函数（Clipped Objective）**: 这是PPO更常用且实现更简单的变体。
    *   **核心思想**: 通过截断（clipping）策略比率 $r_t(\theta) = \frac{\pi_\theta(A_t|S_t)}{\pi_{\theta_k}(A_t|S_t)}$，来限制新旧策略的差异。
    *   **目标函数**:
        $$\mathcal{L}^{CLIP}(\theta) = E_{\tau \sim \pi_k}\left[\sum_{t=0}^T \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)\right]$$
        其中 $\hat{A}_t$ 是优势函数估计，$\epsilon$ 是一个小的截断阈值（例如 $0.1$ 或 $0.2$）。
    *   **工作原理**:
        *   如果优势函数 $\hat{A}_t > 0$（行动是好的），我们希望增加其概率。但如果 $r_t(\theta)$ 变得太大（新策略比旧策略显著增加了这个好行动的概率），截断会限制其增益，防止策略过度激进。
        *   如果优势函数 $\hat{A}_t < 0$（行动是坏的），我们希望降低其概率。但如果 $r_t(\theta)$ 变得太小（新策略显著降低了这个坏行动的概率），截断会限制其损失，防止策略惩罚过重。
        *   本质上，截断目标函数使得策略优化在策略离旧策略太远时变得**悲观**，从而避免了不稳定的巨幅策略变化。
    *   **优点**: 实现简单，经验性能通常与自适应KL惩罚版本相当或更好，不需要显式计算KL散度。


---


## Lecture 7


## 1. 关键要点 (Key Points)

-   **策略梯度问题回顾**: 香草策略梯度（如REINFORCE）存在**高方差**和**样本效率低**（每次更新后需丢弃旧数据）的问题，以及**步长选择困难**。
-   **广义优势函数估计器（Generalized Advantage Estimator, GAE）**:
    *   GAE是一种结合了**多步TD误差**的优势函数估计器，通过一个指数加权平均来平衡TD的低方差和MC的低偏差。
    *   通过参数 $\lambda \in [0,1]$ 平衡偏差和方差：$\lambda=0$ 对应TD(0)误差（低方差，高偏差），$\lambda=1$ 对应MC回报（高方差，低偏差）。
    *   GAE通常在PPO中用作优势函数估计。
-   **策略改进的单调性理论**:
    *   理论保证了在限制策略更新幅度（通过KL散度）的情况下，可以**近似地保证策略性能的单调改进**。这是TRPO和PPO等算法的理论基石。
    *   PPO通过**截断（clipping）**或**KL惩罚**来实现保守的策略更新，从而提高数据效率和稳定性。
-   **模仿学习（Imitation Learning）**:
    *   **动机**: 当奖励函数难以手动设计或强化学习的探索效率低下时，模仿学习提供了一种替代方案：直接从专家示范中学习。
    *   **核心思想**: 学习一个策略，使其行为与专家提供的一组示范轨迹（状态-行动序列）相匹配。
-   **行为克隆（Behavioral Cloning, BC）**:
    *   **方法**: 将模仿学习问题简化为一个标准的**监督学习问题**。将专家示范中的状态-行动对作为训练数据，训练一个分类器（离散行动）或回归器（连续行动），将状态映射到专家行动。
    *   **优点**: 简单、直接，可以利用监督学习的现有工具和理论。
    *   **缺点**:
        1.  **独立同分布（i.i.d.）假设违背**: BC假设训练数据是i.i.d.的，但智能体在环境中行动时，其访问的状态分布会偏离专家示范中的状态分布。
        2.  **复合误差（Compounding Errors）**: 智能体的微小误差在序贯决策中会累积，导致智能体进入训练数据中未见的、偏离专家轨迹的状态，从而无法正确决策，性能急剧下降。误差可能以平方形式增长 $O(\epsilon T^2)$。
-   **DAGGER (Dataset Aggregation)**:
    *   **方法**: 行为克隆的迭代改进算法，旨在解决复合误差问题。它通过**交互式地收集数据**，让智能体在学习过程中探索自己的状态分布，并请求专家对这些新状态下的行动进行标注。
    *   **流程**: 迭代地运行当前策略，收集状态；专家对这些新状态进行标注；将新标注的数据添加到数据集；用聚合后的数据集重新训练策略。
    *   **优点**: 解决了复合误差和数据分布不匹配问题，理论上保证收敛到专家策略。
-   **逆强化学习（Inverse Reinforcement Learning, IRL）**:
    *   **动机**: 不直接学习策略，而是尝试从专家示范中**推断出潜在的奖励函数**，然后在这个推断出的奖励函数上运行标准强化学习算法来学习策略。
    *   **核心挑战**:
        1.  **奖励函数模糊性（Ambiguity）**: 可能存在无限多个奖励函数可以解释同一组最优专家行为。
        2.  **优化**: 推断奖励函数本身是一个复杂的优化问题。
    *   **主要方法**: 最大熵逆强化学习（Maximum Entropy IRL），生成对抗模仿学习（Generative Adversarial Imitation Learning, GAIL）等。

---

## 2. 详细解读 (Detailed Breakdown)

### 2.1 PPO与策略梯度高级主题回顾 (PPO and Advanced Policy Gradients Recap)

*   **PPO的优势**: 克服了香草策略梯度（如REINFORCE）的样本效率低和步长选择困难的问题。
    *   **数据效率提升**: 通过截断（clipping）或KL惩罚，PPO允许从旧策略中采集的数据进行**多步梯度更新**，减少了每次策略更新后重新采集数据的需求。
    *   **单调改进**: PPO的保守策略更新有助于提高单调改进的似然，使其更稳定。
    *   **流行度**: PPO因其高性能、易于实现和广泛应用（包括ChatGPT）而广受欢迎。

*   **广义优势函数估计器 (Generalized Advantage Estimator, GAE)**:
    *   在策略梯度中，优势函数 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 作为梯度更新的权重，能够有效降低方差。GAE提供了一种更优的优势函数估计方式。
    *   **核心思想**: GAE是不同**$k$-步回报优势函数估计**的指数加权平均，它通过一个权重参数 $\lambda \in [0,1]$ 来平衡偏差和方差。
        *   **TD误差**: 定义 $\delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 为一步TD误差。
        *   **$k$-步优势估计**: $A_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V + \gamma^k V(S_{t+k}) - V(S_t)$。
        *   **GAE公式**: GAE通过对这些$k$-步估计进行加权平均得到：
            $$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$
            这个公式可以截断为有限步，例如在PPO中通常运行策略 $T$ 步，然后用一个截断的GAE版本。
    *   **$\lambda$ 的作用**:
        *   当 $\lambda=0$ 时，GAE退化为一步TD误差，具有**最低方差和最高偏差**。
        *   当 $\lambda=1$ 时，GAE退化为蒙特卡洛回报（减去基线），具有**最高方差和最低偏差**。
        *   在实践中，选择 $\lambda \in (0,1)$ 来找到方差与偏差之间的最佳平衡。
    *   **收益**: GAE使得策略梯度方法在平衡学习效率和稳定性方面做得更好。PPO通常使用截断版本的GAE。

### 2.2 模仿学习 (Imitation Learning)

*   **强化学习的挑战**:
    1.  **奖励函数设计（Reward Shaping）**: 在复杂任务中手动设计一个好的奖励函数非常困难，容易导致奖励稀疏或引入错误的行为。
    2.  **探索效率低下**: 智能体需要大量探索才能发现高回报的轨迹，这在现实世界中可能耗时、昂贵甚至危险。
*   **模仿学习的动机**: 当专家（人类或其他智能系统）可以相对容易地**示范期望的行为**时，我们可以通过模仿学习来避免奖励设计和探索难题。
*   **核心思想**: 从专家提供的一组示范轨迹（状态-行动序列）中直接学习一个策略，使其行为与专家相匹配。
*   **问题设置**:
    *   **输入**: 状态空间、行动空间、转换模型（可选）、**没有奖励函数**，以及一个或多个专家示范序列（由专家策略 $\pi^*$ 生成的行动）。
    *   **目标**: 学习一个能够复现专家行为的策略。

#### 2.2.1 行为克隆 (Behavioral Cloning, BC)

*   **方法**: 最简单直接的模仿学习方法。将问题转化为一个标准的**监督学习问题**。
    1.  **数据收集**: 从专家示范中提取状态-行动对 $(s, a)$，作为训练数据集。
    2.  **模型训练**: 训练一个分类器（离散行动）或回归器（连续行动），将状态 $s$ 映射到专家行动 $a$，即学习 $\pi(s) \approx \pi^*(s)$。可以使用神经网络、决策树等模型。
*   **优点**:
    *   **简单易实现**: 直接利用成熟的监督学习技术。
    *   **快速学习**: 可以在静态数据集上进行离线训练。
*   **缺点**:
    1.  **独立同分布（i.i.d.）假设违背**: 监督学习通常假设训练数据和测试数据是i.i.d.的。然而，在序贯决策中，智能体学习的策略 $\pi$ 会影响它访问的状态分布 $d^\pi(s)$。一旦智能体的策略偏离专家策略 $\pi^*$，它可能会进入专家示范中未见过的状态。
    2.  **复合误差（Compounding Errors）**: 这是BC最严重的问题。即使智能体在每个时间步的决策只有微小的误差 $\epsilon$，这些误差在长回合中会累积。智能体一旦偏离专家轨迹，就会进入未训练过的状态空间，从而导致更多、更大的错误，性能迅速崩溃。误差可能以 $O(\epsilon T^2)$ 的形式增长，其中 $T$ 是回合长度。

#### 2.2.2 DAGGER (Dataset Aggregation)

*   **动机**: 解决行为克隆中的复合误差和数据分布不匹配问题。
*   **方法**: DAGGER 是一种**迭代的、交互式**的算法，它**在线**地聚合数据：
    1.  **初始化**: 初始化策略 $\pi_0$（例如，通过行为克隆或随机策略）。初始化空数据集 $\mathcal{D} \leftarrow \emptyset$。
    2.  **循环迭代**:
        *   **运行策略**: 使用当前策略 $\pi_i$ 在环境中运行 $T$ 步，收集轨迹中的状态序列。
        *   **专家标注**: 对于策略 $\pi_i$ 访问过的每个状态 $s_t$，请求专家提供该状态下的正确行动 $a_t^* = \pi^*(s_t)$。
        *   **聚合数据集**: 将这些新的状态-行动对 $(s_t, a_t^*)$ 添加到数据集 $\mathcal{D}$ 中，即 $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s_t, a_t^*)\}$。
        *   **再训练**: 使用聚合后的数据集 $\mathcal{D}$ 重新训练策略 $\pi_{i+1}$。
    3.  **收敛**: 迭代进行，直到策略收敛或达到一定迭代次数。
*   **优点**:
    *   **解决复合误差**: 通过在学习过程中**主动探索**策略所访问的状态，并请求专家在这些新状态下提供反馈，DAGGER确保了训练数据能覆盖智能体实际访问的状态分布。
    *   **理论保证**: DAGGER理论上可以收敛到一个接近专家策略的策略。
    *   **最终策略是确定性的**。
*   **局限性**:
    *   需要**在线访问专家**: 这意味着每次迭代都需要专家提供新的实时标注，这在许多实际应用中可能成本很高或不可行。

#### 2.2.3 逆强化学习 (Inverse Reinforcement Learning, IRL)

*   **动机**: 行为克隆直接模仿行为，但没有理解行为背后的“意图”或“目标”。IRL尝试从专家示范中**推断出潜在的奖励函数 $R$**。一旦推断出奖励函数，就可以使用任何标准RL算法（如Q-learning，策略梯度）来学习一个（理论上）最优的策略。
*   **问题**: 给定状态空间、行动空间、转换模型，以及专家示范，**找到一个奖励函数 $R$**，使得专家示范在这种 $R$ 下是近似最优的。
*   **挑战**:
    1.  **奖励函数模糊性 (Ambiguity)**: 可能存在**无限多个**奖励函数能够使得专家策略是最优的。例如，一个恒定的奖励函数会使所有策略都最优，但它没有区分度。
    2.  **特征匹配**: 如果奖励函数是特征的线性组合 $R(s) = w^T x(s)$，那么IRL的目标是找到权重向量 $w$，使得专家策略下的特征期望与学习到的策略下的特征期望相匹配，从而使专家策略的价值最大化。
        $$E_{\pi^*}[\sum_t \gamma^t x(S_t)] \approx E_\pi[\sum_t \gamma^t x(S_t)]$$
    3.  **优化复杂**: IRL本身是一个复杂的双层优化问题：内层是RL问题（在给定奖励函数下找最优策略），外层是寻找奖励函数以匹配专家行为。
*   **主要方法**:
    *   **最大熵逆强化学习（Maximum Entropy IRL）**: 引入熵的概念来解决奖励模糊性，偏好那些能够解释专家行为但又尽可能随机的策略。
    *   **生成对抗模仿学习（Generative Adversarial Imitation Learning, GAIL）**: 将IRL和模仿学习结合，利用生成对抗网络（GAN）的思想。一个生成器（策略）试图生成像专家一样的行为，一个判别器试图区分专家行为和生成器行为。判别器的输出可以作为奖励信号来训练策略。GAIL是当前最先进的模仿学习方法之一，解决了传统IRL中内层RL问题计算量大的问题。

---
