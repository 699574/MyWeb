---
title: 'RLHF'
date: '2025-07-14'
tags: ['强化学习', 'RLHF']
---


## Lecture 8: Imitation Learning

## 1. 关键要点 (Key Points)

-   **逆强化学习（IRL）的核心挑战**: 从专家示范中推断奖励函数存在**模糊性（Ambiguity）**，即可能存在无限多个奖励函数可以解释同一组最优专家行为。
-   **最大熵逆强化学习（Maximum Entropy IRL）**:
    *   **原理**: 旨在解决奖励函数模糊性。在所有能够使得专家示范最优的奖励函数中，它选择那个能够使得策略（轨迹分布）**熵最大**的奖励函数。
    *   **优势**: 偏好那些能够解释专家行为但又尽可能随机（不确定性最大）的策略，这有助于避免过拟合专家示范，并允许在示范不足的情况下进行泛化。
    *   **实现**: 通常通过迭代优化奖励函数和策略来实现，需要环境动力学模型知识。
-   **模仿学习总结**:
    *   能够显著减少学习一个良好策略所需的数据量。
    *   一个激动人心的研究方向是结合逆强化学习/从示范中学习与在线强化学习。
-   **人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）**:
    *   **动机**: 传统RL的奖励函数设计困难，或者目标是使AI行为与人类价值观/偏好对齐。
    *   **核心思想**: 不直接手动设计奖励函数，而是**从人类反馈中学习一个奖励模型**，然后用这个学习到的奖励模型来训练RL智能体。
    *   **RLHF流程（以ChatGPT为例）**:
        1.  **监督微调（Supervised Fine-Tuning, SFT）**: 首先在大规模文本数据上进行预训练，然后用高质量的人类编写的指令-响应对进行监督微调，使模型能够遵循指令。
        2.  **训练奖励模型（Reward Model, RM）**:
            *   模型生成多个（例如4个）响应。
            *   人类标注员对这些响应进行**成对比较（Pairwise Comparisons）**，表达对不同响应的偏好（例如，哪个更好，哪个更差）。
            *   根据这些比较数据训练一个奖励模型（通常是另一个神经网络），使其能够预测人类对任何给定响应的偏好程度，从而输出一个标量奖励值。常用的模型如Bradley-Terry模型。
        3.  **策略优化（Policy Optimization）**: 使用训练好的奖励模型作为环境的奖励函数，利用**近端策略优化（PPO）**算法来微调语言模型。
            *   奖励函数是 **$R(s) = RM_\phi(s) - \beta \log \frac{P_{RL}(s)}{P_{PT}(s)}$**，其中 $RM_\phi(s)$ 是奖励模型给出的奖励，后一项是**KL散度惩罚**，防止RL训练导致策略偏离预训练模型太远，确保生成质量和多样性。
-   **成对比较的优势**: 人类通常更容易对两个或多个选项进行**相对比较**（哪个更好），而不是直接给出精确的**标量评分**（比如“这个输出值10分”）。这种方式收集的反馈更可靠且易于标注。
-   **RLHF的成果**: ChatGPT等大型语言模型的出色表现，正是RLHF结合PPO和高质量数据的结果。它显著提升了模型遵循指令、避免有害输出和与人类意图对齐的能力。

---

## 2. 详细解读 (Detailed Breakdown)

### 2.1 逆强化学习（IRL）的深度探讨 (Deeper Dive into Inverse Reinforcement Learning)

*   **回顾**: IRL的目标是根据专家示范来推断出奖励函数。
*   **奖励模糊性**: 这是IRL的核心挑战。存在**无限多个**奖励函数都可以使得给定的专家策略是最优的。例如，一个常量奖励函数会让所有策略都“最优”，但它不具区分度。
*   **最大熵逆强化学习（Maximum Entropy IRL）**:
    *   **核心原理**: 在所有能够使得专家示范最优的奖励函数中，最大熵IRL选择那个能使得由该奖励函数导出的**策略（或轨迹分布）熵最大**的奖励函数。
    *   **熵的最大化**: 熵最大化对应着不确定性最大化。在给定观测数据的约束下，最大熵原理认为，最能代表我们当前知识的概率分布是熵最大的那个。
    *   **IRL中的应用**: 将这一原理应用于IRL意味着，在所有能解释专家行为的奖励函数中，我们选择那个导出的策略（或轨迹分布）最“不确定”的奖励函数。这有助于避免过拟合专家示范的细节，使得学习到的策略在示范数据稀疏的区域能更好地泛化。
    *   **数学形式**:
        $$\max_R - \sum_\tau P(\tau) \log P(\tau) \quad \text{s.t.} \quad E_P[\mu(\tau)] = E_{\text{expert}}[\mu(\tau)]$$
        其中 $P(\tau)$ 是由奖励函数 $R$ 决定的轨迹分布，$\mu(\tau)$ 是轨迹的特征（通常是特征向量的和）。目标是找到奖励函数参数，使得模型生成的轨迹的特征期望与专家示范的特征期望相匹配。
    *   **实现**: 通常涉及迭代过程：
        1.  **初始化奖励模型**。
        2.  **给定奖励模型，使用标准RL算法计算最优策略**（这一步需要知道动力学模型）。
        3.  **计算新策略下的状态访问频率或特征期望**。
        4.  **根据专家示范和新策略的特征期望，更新奖励模型的参数**，以最大化专家示范在该奖励模型下的似然，并使模型更接近专家行为。
    *   **挑战**: 原始的最大熵IRL算法通常需要**已知环境动力学模型**，并且在每一步迭代中都需要解决一个完整的强化学习问题（计算最优策略），这在计算上可能非常昂贵。后续工作（如Guided Cost Learning by Finn et al. 2016）解决了无需已知动力学模型的问题，并使用了更通用的奖励/成本函数。

### 2.2 模仿学习总结 (Imitation Learning Summary)

*   **减少数据需求**: 模仿学习能够显著减少学习一个良好策略所需的数据量，因为它利用了专家提供的高质量先验知识。
*   **挑战与机遇**: 模仿学习领域仍在发展，一个重要的研究方向是结合逆强化学习/从示范中学习与在线强化学习，以进一步提高效率和性能。

### 2.3 人类反馈与偏好学习（RLHF） (Human Feedback and Reinforcement Learning from Human Preferences)

*   **核心动机**: 在许多复杂任务中，很难手动定义一个精确的奖励函数来捕捉人类对“好”行为的复杂、主观偏好（例如，生成一个有帮助、无害、有创意且真实的文本）。
*   **基本理念**: 不直接编写奖励函数，而是**从人类对模型行为的反馈中学习一个奖励模型**，然后用这个学习到的奖励模型作为RL训练的替代奖励信号。
*   **历史与发展**:
    *   早期的尝试（如TAMER框架）直接从人类的即时反馈中学习奖励模型。
    *   更近期的工作，特别是Christiano et al. (2017) 的研究，证明了通过收集人类对**轨迹或响应片段的成对比较**来学习奖励模型的有效性。这种方式比直接给出标量奖励更容易让人类操作员进行。
    *   这种方法在训练机器人后空翻等任务中取得了成功，仅需几百次人类偏好反馈。
*   **RLHF流程详解（以大型语言模型为例，如InstructGPT/ChatGPT）**:
    1.  **监督微调（Supervised Fine-Tuning, SFT）**:
        *   **目的**: 使预训练的大型语言模型能够遵循指令并产生高质量的输出。
        *   **方法**: 收集少量高质量的人类编写的指令-响应示范（例如，人类撰写的答案或人类对模型输出的修改），然后用这些数据对预训练模型进行监督学习（微调）。
    2.  **训练奖励模型（Reward Model, RM）**:
        *   **目的**: 学习一个能够量化人类偏好的奖励函数。
        *   **数据收集**:
            *   用SFT后的语言模型对一个给定指令生成多个不同的响应（例如，2-9个响应）。
            *   人类标注员对这些响应进行**排序**或**成对比较**（例如，“哪个响应更好？”）。这种相对比较比直接评分更容易、更可靠。
            *   积累大量这样的比较数据。
        *   **模型训练**: 训练一个单独的神经网络（奖励模型RM），其输入是指令和模型响应，输出是一个**标量奖励值**。训练目标是使RM能够准确预测人类的偏好。通常使用Bradley-Terry模型或其变体作为损失函数来优化RM。
    3.  **策略优化（Policy Optimization）**:
        *   **目的**: 微调语言模型，使其能够最大化由奖励模型给出的奖励，从而生成更符合人类偏好的输出。
        *   **算法**: 通常使用**近端策略优化（PPO）**算法。
        *   **奖励函数**: 在PPO的优化目标中，除了奖励模型给出的奖励 $RM_\phi(s)$，通常还会添加一个**KL散度惩罚项**：
            $$R_{RL}(s) = RM_\phi(s) - \beta D_{KL}(P_{RL}(s) || P_{PT}(s))$$
            其中 $P_{RL}(s)$ 是RL训练后的模型策略， $P_{PT}(s)$ 是SFT阶段的监督微调模型（或原始预训练模型）的策略。
            *   **KL惩罚的作用**: 防止模型在优化奖励的同时，过度偏离原始的预训练模型。这有助于保持模型的泛化能力、连贯性和避免生成不合理的输出。$\beta$ 是一个超参数，用于平衡奖励优化和策略保持。
*   **RLHF的成果**:
    *   在人类评估中，RLHF训练的模型（如InstructGPT/ChatGPT）显著优于单纯的预训练和监督微调模型。
    *   它使得大型语言模型能够更好地遵循指令、生成更少有害内容、更符合人类价值观和意图的输出。
    *   尽管有更简单的基线方法（如“Best-of-n”选择最佳输出），但PPO等RLHF方法确实能带来进一步的性能提升。

---
## Lecture 9

## 1. 关键要点 (Key Points)

-   **RLHF回顾**: RLHF通过三个主要阶段使AI系统与人类意图对齐：
    1.  **监督微调（SFT）**: 基于高质量指令-响应对的模型初始化。
    2.  **奖励模型（RM）训练**: 从人类对模型响应的**成对比较**中学习一个量化人类偏好的奖励函数。
    3.  **策略优化（PPO）**: 使用学习到的奖励模型作为奖励信号，通过强化学习（通常是PPO）微调SFT模型。
-   **成对比较的优势**: 人类更擅长对两个选项进行相对偏好比较，而不是给出绝对评分，这种数据收集方式更可靠。
-   **Bradley-Terry模型**: 在奖励模型训练中，Bradley-Terry模型常用于将成对比较转化为不同选项背后潜在奖励的概率模型。其核心公式为：$P(b_i \succ b_j) = \frac{\exp(r(b_i))}{\exp(r(b_i)) + \exp(r(b_j))}$，其中 $r(b_i)$ 是选项 $b_i$ 的潜在奖励。
-   **奖励函数的模糊性与不变性**: 奖励函数可以被任意常数平移和任意正数缩放，而不会改变其导出的最优策略或偏好顺序。这意味着存在无限多组奖励参数能产生相同的最优策略，这给IRL带来了挑战。
-   **RLHF的优化目标**: 在策略优化阶段，RLHF的目标是最大化学习到的奖励模型给出的回报，同时通过KL散度惩罚项约束策略不要偏离参考模型太远，以保持生成质量和多样性。
    $$\max_{\pi_\theta} E_{x \sim D, y \sim \pi_\theta(y|x)} [r_\phi(x, y)] - \beta D_{KL}(\pi_\theta(\cdot|x)||\pi_{ref}(\cdot|x))$$
    其中 $r_\phi$ 是奖励模型，$\pi_{ref}$ 是参考策略。
-   **直接偏好优化（Direct Preference Optimization, DPO）**:
    *   **动机**: 简化RLHF流程，去除显式奖励模型训练和PPO强化学习的复杂性。
    *   **核心思想**: 发现上述RLHF优化目标在特定奖励函数（通常是策略概率的对数比）下存在**闭式最优策略（Closed-Form Optimal Policy）**。DPO利用这一性质，将奖励模型训练和策略优化**整合**到一步中。
    *   **推导简述**:
        1.  对于RLHF目标函数，其最优策略 $\pi^*(y|x)$ 可以表示为参考策略 $\pi_{ref}(y|x)$ 和奖励函数 $r(x,y)$ 的函数：
            $$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$
            其中 $Z(x)$ 是归一化常数。
        2.  将这个闭式形式的策略代入Bradley-Terry偏好模型，可以得到：
            $$\log \sigma(r(x, y_w) - r(x, y_l))$$
            将其中的 $r(x,y)$ 用 $\beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$ 替换（这是由上述闭式形式倒推出的等价关系）。
        3.  最终的DPO损失函数可以直接优化策略参数 $\theta$，而无需中间的奖励模型 $r$：
            $$L_{DPO}(\pi_\theta; \pi_{ref}) = -E_{(x,y_w,y_l) \sim D} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$
            这个损失函数直接以人类偏好数据 $(x,y_w,y_l)$ 为输入，通过策略概率比的对数差来训练策略，从而实现与人类偏好的对齐。
    *   **优点**:
        1.  **简化**: 省去了奖励模型训练和PPO训练的两个复杂阶段，降低了实现难度。
        2.  **稳定**: 避免了奖励模型的过拟合，以及PPO训练中的复杂超参数调整和“致命三元组”问题。
        3.  **高效**: 经验上性能与RLHF相当或更好，尤其在大型语言模型上表现卓越。
-   **大规模应用**: DPO在大型语言模型（如Mistral、LLaMa3）的指令微调中取得了巨大成功，甚至在MT-Bench等基准测试中超越了许多传统RLHF训练的模型。

---

## 2. 详细解读 (Detailed Breakdown)

### 2.1 RLHF策略优化回顾 (RLHF Policy Optimization Recap)

*   **RLHF管道**: RLHF是一个三阶段管道：监督微调（SFT）得到初始语言模型；从人类偏好数据中训练奖励模型（RM）；使用PPO或其他RL算法在RM上微调语言模型。
*   **奖励模型训练**:
    *   核心在于从人类的**成对比较**中学习一个能够反映人类偏好的奖励函数 $r_\phi(x,y)$。
    *   **Bradley-Terry模型**是这一过程的数学基础，它假设人类偏好 $y_w$ 优于 $y_l$ 的概率与它们潜在奖励的指数比相关。
    *   训练奖励模型通过最大化人类偏好数据的似然来完成，通常是最小化负对数似然损失：
        $$L_R(\phi, D) = -E_{(x,y_w,y_l) \sim D} [\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))]$$
        其中 $D$ 是成对偏好数据集，$\sigma$ 是sigmoid函数，$r_\phi$ 是参数为 $\phi$ 的奖励模型。
*   **策略优化**:
    *   奖励模型训练完成后，它被用作RL环境的奖励函数。
    *   语言模型通过PPO进行优化，目标是最大化奖励模型给出的回报，同时通过KL散度惩罚项防止策略过度偏离原始模型。
    *   目标函数: $\max_{\pi_\theta} E[r_\phi(x,y)] - \beta D_{KL}(\pi_\theta||\pi_{ref})$。

### 2.2 直接偏好优化 (Direct Preference Optimization, DPO)

DPO的核心创新在于，它意识到RLHF的优化目标在特定条件下存在一个**闭式最优策略**，这使得整个三阶段流程可以被大幅简化。

*   **DPO的推导核心**:
    1.  **RLHF优化目标**: 回顾标准的RLHF优化目标，即最大化奖励并约束KL散度。
    2.  **闭式最优策略**: 对于形如 $\max_\pi E[r(x,y)] - \beta D_{KL}(\pi||\pi_{ref})$ 的优化问题，存在一个已知的闭式最优策略 $\pi^*(y|x)$，它是一个 Boltzmann 分布：
        $$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$
        其中 $Z(x)$ 是归一化常数。
    3.  **逆向工程**: DPO的关键洞察是“逆向工程”这个关系。如果一个策略 $\pi$ 是这个形式的最优策略，那么它“隐式地”定义了一个奖励函数 $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$。由于奖励函数可以被常数平移而不改变策略或偏好，所以我们可以简化为 $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$。
    4.  **构建DPO损失**: 将这个“隐式”奖励函数代入Bradley-Terry模型的损失函数（即RLHF中奖励模型训练的损失函数）：
        $$L_{DPO}(\pi_\theta; \pi_{ref}) = -E_{(x,y_w,y_l) \sim D} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$
        *   这个DPO损失函数直接使用策略 $\pi_\theta$（以及参考策略 $\pi_{ref}$）来计算比较项，而不再需要一个独立的奖励模型。
        *   通过最小化这个损失函数，我们直接优化策略 $\pi_\theta$，使其生成人类偏好度更高的响应。
*   **DPO的优势**:
    1.  **简化流程**: 将奖励模型训练和强化学习优化合二为一，不再需要独立的奖励模型网络和复杂的PPO算法。
    2.  **稳定高效**: 避免了奖励模型训练可能出现的过拟合问题，以及PPO训练中对超参数（如KL系数、优势函数估计器）的敏感性，从而使得训练更稳定、样本效率更高。
    3.  **梯度直接**: DPO的损失函数直接对策略参数 $\theta$ 产生梯度，可以直接通过标准的反向传播和优化器进行训练。

### 2.3 大规模DPO训练与成果 (Large-Scale DPO Training and Results)

*   DPO已在大型语言模型（LLM）的指令微调中得到广泛应用，例如Meta的LLaMa3、Mistral Instruct等。
*   **性能表现**: 在各种基准测试（如MT-Bench）中，DPO训练的模型在人类偏好方面表现出色，甚至能够超越许多通过传统RLHF（PPO+RM）训练的模型。
*   **对齐的含义**: DPO及RLHF的成功，意味着AI系统能够更好地理解并遵循人类的指令，生成符合人类价值观和意图的响应，这正是AI“对齐”的核心目标。



## Lecture 10: Offline RL

## 1. 关键要点 (Key Points)

-   **离线强化学习（Offline RL）**:
    *   **定义**: 仅从一个预先收集的固定数据集 $\mathcal{D}$ 中学习策略，**不与环境进行新的交互**。
    *   **动机**: 在线RL的探索成本过高或不安全（如医疗、教育、机器人控制、推荐系统），数据已存在且无需额外收集。
    *   **目标**: 从历史决策中学习，避免未来潜在的糟糕决策。
-   **离线策略评估（Offline Policy Evaluation, OPE）**: 估计一个新策略 $\pi$ 在特定历史数据集 $\mathcal{D}$ 上的表现。
    *   **基于模型**: 从 $\mathcal{D}$ 学习环境模型 $\hat{P}, \hat{R}$，然后在该模型上进行策略评估。可能存在**模型误指定偏差（Bias due to Model Misspecification）**。
    *   **无模型**:
        *   **重要性采样（Importance Sampling, IS）**: OPE的核心方法。通过对行为策略 $\pi_b$（收集数据的策略）产生的样本赋予权重，来估计目标策略 $\pi$ 的期望回报。
            $$E_p[r] = E_q[\frac{p(x)}{q(x)} r(x)]$$
            其中 $\frac{p(x)}{q(x)}$ 是重要性权重。
        *   **优点**: 能够对任意目标策略进行无偏估计（如果**数据覆盖**充分）。
        *   **缺点**: **高方差**是其主要挑战，尤其是在长期序贯决策或行为策略与目标策略差异大时。重要性权重可能爆炸或消失。
        *   **Per-Decision Importance Sampling (PDIS)**: 结合了时序结构，将重要性权重分解到每一步。
-   **数据覆盖（Overlap）要求**: 为了进行无偏的离线评估或优化，行为策略 $\pi_b$ 必须在所有目标策略 $\pi$ 可能采取行动的状态-行动对上提供**非零的支持（non-zero support）**。即，如果目标策略在某个状态-行动对 $(s,a)$ 下的概率 $\pi(a|s)>0$，则行为策略在该对下的概率 $\pi_b(a|s)$ 也必须大于 $0$。若无覆盖，则无法进行准确评估。
-   **离线策略优化（Offline Policy Optimization, OPO）**:
    *   **定义**: 从固定数据集 $\mathcal{D}$ 中学习一个**最优或接近最优**的策略 $\pi$。
    *   **挑战**:
        1.  **数据覆盖不足/外推误差**: 智能体倾向于选择数据集中没有充分支持的行动，导致对其性能的**过高估计**，部署时可能表现糟糕。
        2.  **“浓缩性”假设（Concentrability Assumption）**: 许多传统算法依赖于数据集中包含对所有可能策略的充分支持，但在现实中这很少成立。
-   **保守离线RL（Conservative Offline RL）**:
    *   **核心思想**: 在数据支持不足（即数据覆盖率低）的状态-行动空间区域，算法采取**悲观主义（Pessimism）**原则，假设这些区域的价值很低。
    *   **方法**: 通过**惩罚模型不确定性**或**限制策略在外推区域的探索**来确保学习的策略是安全的、可部署的。
    *   **例如**: Conservative Q-Learning (CQL) 等方法。
    *   **目标**: 在数据覆盖不全的情况下，找到一个在**支持区域内表现最佳**的策略。

---

## 2. 详细解读 (Detailed Breakdown)

### 2.1 离线强化学习的背景与动机 (Introduction and Setting of Offline RL)

*   **在线RL的局限**: 传统的在线强化学习需要智能体持续与环境进行交互和探索。但在许多现实场景中，这种交互是：
    *   **成本高昂**: 例如，在医疗、教育、大规模A/B测试中。
    *   **不安全**: 例如，自动驾驶、机器人控制。
    *   **不可行**: 例如，历史日志数据分析，无法重复试验。
*   **离线RL的诞生**: 为了应对这些挑战，离线RL应运而生。它专注于在**完全不进行新的环境交互**的情况下，仅从一个预先收集的静态数据集 $\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_i$ 中学习最优策略。
*   **目标**: 从过去的决策和结果中学习，目的是识别并学习一个能够带来比过去更好的未来结果的策略。这是一种“事后诸葛亮”的学习，但其目标是为未来提供更好的决策。

### 2.2 离线批量策略评估 (Offline Batch Policy Evaluation, OPE)

OPE的目标是估计一个给定策略 $\pi$ 在一个固定数据集 $\mathcal{D}$ 上的预期表现。

#### 2.2.1 基于模型的方法 (Using Models)

*   **方法**: 首先从给定的数据集 $\mathcal{D}$ 中学习一个环境模型（即转换模型 $\hat{P}(s'|s,a)$ 和奖励模型 $\hat{R}(s,a)$），然后在这个学习到的模型上，使用动态规划（如策略迭代或价值迭代）来评估目标策略 $\pi$ 的价值。
*   **优点**: 如果模型准确，可以得到目标策略的良好估计。
*   **缺点**:
    *   **模型误指定（Model Misspecification）**: 如果学习到的模型与真实环境不符，评估结果将存在偏差。
    *   **模型不确定性**: 在数据稀疏的区域，学习到的模型可能非常不准确，导致评估结果不可靠。

#### 2.2.2 无模型方法 (Using Model-Free Methods)

*   **挑战**: 传统的无模型方法（如MC和TD）通常是同策略的，或者在异策略学习时需要环境交互。在离线设置中，我们无法再进行新的交互来修正数据分布。
*   **解决方案**: **重要性采样（Importance Sampling, IS）**。

#### 2.2.3 使用重要性采样 (Using Importance Sampling)

*   **原理**: 重要性采样是一种统计学技术，用于估计一个分布下的期望，通过从另一个不同的分布中采样来实现。在OPE中，我们想估计目标策略 $\pi$ 下的预期回报 $E_\pi[R(\tau)]$，但我们只有行为策略 $\pi_b$ 下的样本。IS通过给每个样本赋予一个权重 $\omega(\tau) = \frac{P_\pi(\tau)}{P_{\pi_b}(\tau)}$ 来修正分布差异。
    $$E_\pi[R(\tau)] = E_{\pi_b}\left[\frac{P_\pi(\tau)}{P_{\pi_b}(\tau)} R(\tau)\right] \approx \frac{1}{N} \sum_{i=1}^N \frac{P_\pi(\tau^{(i)})}{P_{\pi_b}(\tau^{(i)})} R(\tau^{(i)})$$
    其中，轨迹的概率比可以分解为每一步行动的概率比的乘积：$\frac{P_\pi(\tau)}{P_{\pi_b}(\tau)} = \prod_{t=0}^{T-1} \frac{\pi(A_t|S_t)}{\pi_b(A_t|S_t)}$。
*   **优点**:
    *   **无偏估计**: 如果行为策略 $\pi_b$ 对所有目标策略 $\pi$ 可能采取的行动都具有非零支持（即 $\pi_b(a|s)>0$ 当 $\pi(a|s)>0$ 时，这个条件称为**覆盖（Coverage）或重叠（Overlap）**），那么重要性采样可以提供目标策略价值的无偏估计。
    *   **无需模型**: 直接从数据中学习，不需要环境模型。
*   **缺点**:
    *   **高方差**: 这是IS的主要挑战。当目标策略 $\pi$ 和行为策略 $\pi_b$ 差异较大，或者回合长度 $T$ 较长时，重要性权重 $\prod \frac{\pi(A_t|S_t)}{\pi_b(A_t|S_t)}$ 可能变得非常大或非常小（**权重爆炸或消失**），导致估计方差巨大，评估结果不可靠。
    *   **覆盖要求**: 必须满足覆盖条件。如果目标策略 $\pi$ 采取了行为策略 $\pi_b$ 从未采取过或极少采取的行动，IS的估计将非常不稳定甚至不可能。
    *   **Per-Decision Importance Sampling (PDIS)**: 为了缓解长回合中的方差问题，PDIS利用领域内的时序结构，将重要性权重分解到每个决策点上，但在本质上仍然面临方差问题。

### 2.3 离线策略学习/优化 (Offline Policy Learning / Optimization, OPO)

OPO的目标是直接从固定数据集中找到一个最优或接近最优的策略。

*   **挑战：数据分布不匹配/外推误差**:
    *   在在线RL中，智能体可以探索并收集它所访问的状态-行动对数据。但在离线RL中，智能体只能从行为策略 $\pi_b$ 生成的数据中学习。
    *   如果学习到的策略 $\pi$ 试图采取数据集中**没有充分支持**（即 $\pi_b$ 很少访问）的行动，对其价值的估计将非常不准确。这种现象被称为**外推误差（extrapolation error）**。
    *   更糟糕的是，RL算法往往会**高估**这些未被充分探索的行动的价值，因为它们没有负面反馈，从而导致学习到的策略在实际部署时表现糟糕（**性能崩溃**）。

*   **解决方案：悲观主义（Pessimism）原则**:
    *   **核心思想**: 在数据支持不足（低覆盖率）的区域，我们应该采取**保守（Pessimistic）**的态度，假设这些区域的价值较低。这可以防止策略过度自信地探索未知区域。
    *   **实现方法**:
        1.  **惩罚不确定性**: 在价值函数优化目标中增加惩罚项，当某个状态-行动对的价值估计不确定性高时（因为数据支持少），则降低其估计值。
        2.  **限制策略**: 显式地限制学习到的策略 $\pi$ 只能在数据集中有充分支持的区域内进行动作。
        3.  **Filtration Function**: 引入一个过滤函数 $\zeta(s,a)$ 来指示哪些状态-行动对是“支持良好的”。例如，$\zeta(s,a) = 1$ 如果 $(s,a)$ 被充分观察到，否则为 $0$。
        4.  **贝尔曼操作符的修改**: 修改贝尔曼操作符，使得在数据支持不足的区域，其下一个状态的价值被悲观地估计为0（或其他下界），从而传播悲观主义。
            $$T_f(s,a) = r(s,a) + \gamma E_{s'}[ \max_{a'} \zeta(s',a')f(s',a')]$$
            这里，如果 $(s',a')$ 没有充分数据，$\zeta(s',a')$ 为 $0$，其贡献为 $0$，导致悲观估计。
    *   **算法实例**: **保守Q学习（Conservative Q-Learning, CQL）**是这一方向的代表算法，通过在优化目标中加入惩罚项，鼓励Q函数在数据覆盖不足的区域给出较低的估计。
*   **成果**: 悲观主义方法在实际中表现良好，能够在有限数据集上学习到安全且性能可靠的策略，尤其在医疗、机器人等高风险领域具有巨大潜力。

### 2.4 总结 (Conclusion)

-   **离线RL的重要性**: 解决了在线RL在现实世界部署中的关键挑战，使得RL能够从现有的大规模静态数据中学习。
-   **OPE和OPO的挑战**: 核心挑战是数据覆盖不足导致的外推误差和高方差。
-   **重要性采样**: 是OPE的基础，但其高方差限制了在长期或高维度问题中的应用。
-   **悲观主义原则**: 离线RL应对不确定性的核心策略，通过保守估计未探索区域的价值，提高学习策略的鲁棒性和安全性。
-   **应用前景**: 离线RL在医疗（如个性化胰岛素管理）、教育、推荐系统等领域具有巨大的应用潜力，可以从历史数据中提取可操作的洞察和策略。

---

## Lecture 11: Data Efficient RL

## 1. 关键要点 (Key Points)

-   **数据效率的重要性**: 在许多现实世界的强化学习应用中，与环境的交互（数据收集）是昂贵、耗时或不安全的，因此需要算法在有限数据下高效学习。
-   **多臂老虎机（Multi-armed Bandits, MAB）**:
    *   **定义**: 一个简化的序贯决策问题，智能体在每个时间步从一组“臂”（行动）中选择一个，并获得一个来自该臂的未知概率分布的奖励。
    *   **目标**: 最大化累积奖励（长期回报）。
-   **评估算法性能的指标：后悔（Regret）**:
    *   **定义**: 智能体在给定时间步内所获得的累积奖励与最优策略能够获得的累积奖励之间的差距。
    *   **一步后悔 $l_t = E[V^* - Q(A_t)]$**: 单个时间步的机会损失。
    *   **总后悔 $L_t = \sum_{\tau=1}^t l_\tau$**: 累积的机会损失。
    *   **目标**: 最小化总后悔。理想情况是实现**次线性（Sublinear）后悔**（例如 $O(\log T)$），而不是线性后悔 $O(T)$。
-   **探索与利用的权衡**:
    *   **探索（Exploration）**: 尝试新的行动，以发现潜在的高回报。
    *   **利用（Exploitation）**: 选择当前已知回报最高的行动，以最大化即时奖励。
    *   数据高效学习的关键在于如何智能地平衡这两者。
-   **探索策略**:
    *   **贪婪算法（Greedy Algorithm）**: 总是选择当前估计回报最高的行动。
        *   **缺点**: 容易陷入次优（Suboptimal）行动，一旦某个次优行动被偶然高估，智能体可能永远不再探索其他最优行动。导致**线性后悔**。
    *   **$\epsilon$-贪婪算法（$\epsilon$-Greedy Algorithm）**: 以 $1-\epsilon$ 的概率利用当前最佳行动，以 $\epsilon$ 的概率随机探索。
        *   **缺点**: 如果 $\epsilon$ 固定，即使最优行动已知，也会持续进行随机探索，导致**线性后悔**。
        *   **改进**: 逐渐衰减 $\epsilon$（例如 $\epsilon_t = 1/t$）可以实现次线性后悔。
    *   **乐观面对不确定性（Optimism in the Face of Uncertainty, OFU）原则**: 鼓励智能体探索那些“可能”有高回报的行动。
        *   **UCB1算法（Upper Confidence Bound 1）**: 经典实现。选择 $A_t = \arg\max_a \left[ \hat{Q}_t(a) + c \sqrt{\frac{\log t}{N_t(a)}} \right]$。
            *   $\hat{Q}_t(a)$ 是平均回报估计，$N_t(a)$ 是行动选择次数。
            *   $c \sqrt{\frac{\log t}{N_t(a)}}$ 是**探索奖励项**，它对被探索较少（$N_t(a)$ 小）或时间步较长（$\log t$ 大）的行动给予额外奖励。
        *   **优点**: 在MAB问题中实现**对数（Logarithmic）后悔** $O(\log T)$，这是一种很好的次线性后悔。
    *   **概率匹配 / 汤普森采样（Probability Matching / Thompson Sampling）**:
        *   **原理**: 维护每个行动奖励分布的后验信念（Prior belief）。在每个时间步，从每个行动的后验分布中采样一个可能的价值，然后选择采样价值最高的行动。
        *   **优点**: 经验表现通常与UCB算法相当甚至更好，被认为是实践中非常有效的启发式方法，也能实现次线性后悔。

---

## 2. 详细解读 (Detailed Breakdown)

### 2.1 数据效率与评估标准 (Data Efficiency and Evaluation Criteria)

*   **强化学习的挑战**: 传统的强化学习算法往往需要大量的环境交互，这在许多真实应用中是不可行的。
*   **数据高效RL的目标**: 训练一个能够在最少经验下学习到良好策略的智能体。
*   **评估标准**:
    *   **收敛性**: 算法是否能收敛到最优策略。
    *   **收敛速度**: 达到最优策略所需的时间/迭代次数。
    *   **犯错成本**: 在学习过程中犯错的次数或带来的损失。
*   **多臂老虎机（MAB）作为研究平台**: MAB是序贯决策问题的最简单形式，没有状态转移，每个决策只影响当前奖励，便于分析探索与利用问题。

### 2.2 多臂老虎机 (Multi-armed Bandits)

*   **定义**: 一个MAB问题由一组已知行动 $A$ （臂的数量 $m$）和一个未知奖励分布 $R^a(r)$ 组成，智能体在每个时间步选择一个行动 $A_t$，获得一个来自 $R^{A_t}$ 的奖励 $R_t$。
*   **目标**: 在 $T$ 个时间步内最大化累积奖励 $\sum_{\tau=1}^T R_\tau$。
*   **例子**: “治疗断裂的脚趾”问题：3种治疗方案（行动），每种方案的效果（奖励）未知（二进制变量：愈合或未愈合）。

### 2.3 性能评估框架：后悔 (Regret Framework)

*   **行动价值 $Q(a)$**: 采取行动 $a$ 的期望平均奖励 $E[r|a]$。
*   **最优价值 $V^*$**: 所有行动价值中的最大值 $V^* = \max_a Q(a)$。
*   **一步后悔 $l_t$**: 在时间步 $t$ 选择行动 $A_t$ 所带来的机会损失。
    $$l_t = E[V^* - Q(A_t)]$$
*   **总后悔 $L_t$**: 在总时间步 $T$ 内累积的机会损失。
    $$L_t = \sum_{\tau=1}^t l_\tau$$
*   **理想目标**: 目标是最小化总后悔。如果 $L_t$ 随着时间线性增长 $O(T)$，则表示算法在持续犯错。如果 $L_t$ 增长慢于时间步数（例如 $O(\log T)$），则表示算法随着时间推移能够收敛到最优行动，并且犯错的频率越来越低。
*   **后悔与差距**: 总后悔是每个行动被选择的次数 $N_t(a)$ 与该行动和最优行动之间的差距 $\Delta_a = V^* - Q(a)$ 的函数。一个好的算法会确保“高差距”的行动被选择的次数少。
*   **下界定理（Lai and Robbins）**: 对于任何算法，在MAB问题中的渐近总后悔至少是时间步 $T$ 的对数函数 $O(\log T)$。这个下界由最优行动与次优行动之间的差距以及奖励分布的KL散度决定，表明次线性后悔是理论上可达的。

### 2.4 探索策略与算法 (Exploration Strategies and Algorithms)

#### 2.4.1 贪婪算法 (Greedy Algorithm)

*   **方法**: 每次总是选择当前平均奖励估计值最高的行动。
*   **缺点**: 极容易陷入**局部最优**。如果一个次优行动偶然被高估（例如，因为早期的几次尝试带来了高随机奖励），贪婪算法可能会一直选择它，从而永远无法发现真正的最优行动。
*   **后悔**: 导致**线性后悔 $O(T)$**。

#### 2.4.2 $\epsilon$-贪婪算法 ($\epsilon$-Greedy Algorithm)

*   **方法**: 在每个时间步：
    *   以 $1-\epsilon$ 的概率，选择当前平均奖励估计最高的行动（利用）。
    *   以 $\epsilon$ 的概率，随机选择一个行动（探索）。
*   **优点**: 简单，保证了所有行动都有被探索的机会。
*   **缺点**: 如果 $\epsilon$ 固定，即使已经发现最优行动，智能体也会持续进行随机探索，导致**线性后悔**。因为总会有 $\epsilon$ 比例的时间选择次优行动。
*   **改进**: **衰减 $\epsilon$**：让 $\epsilon$ 随着时间逐渐减小，例如 $\epsilon_t = 1/t$。这样，算法在早期阶段进行更多探索，后期则更多利用，最终收敛到最优行动，从而实现**次线性后悔**（如 $O(\log T)$）。

#### 2.4.3 乐观面对不确定性 (Optimism in the Face of Uncertainty, OFU)

*   **核心思想**: 这种原则鼓励智能体优先探索那些“可能”具有最高价值的行动。它不是盲目地随机探索，而是有目的地探索那些价值不确定但潜力最大的行动。
*   **UCB1算法（Upper Confidence Bound 1）**: 是OFU原则的经典实现。它为每个行动计算一个**置信区间上限**，并选择上限最高的行动。
    *   **选择规则**:
        $$A_t = \arg\max_a \left[ \hat{Q}_t(a) + c \sqrt{\frac{\log t}{N_t(a)}} \right]$$
        其中：
        *   $\hat{Q}_t(a)$ 是行动 $a$ 的平均奖励估计。
        *   $N_t(a)$ 是行动 $a$ 被选择的次数。
        *   $c \sqrt{\frac{\log t}{N_t(a)}}$ 是**探索奖励项**。这个项：
            *   随着时间 $t$ 的增加而增加（鼓励总体探索）。
            *   随着行动 $a$ 被选择次数 $N_t(a)$ 的增加而减小（被探索过的行动，其不确定性降低，探索奖励减小）。
            *   这意味着UCB1会优先选择那些：1) 估计平均奖励高，或 2) 被探索次数少从而不确定性高 的行动。
    *   **理论基础**: UCB1的探索奖励项来源于**霍夫丁不等式（Hoeffding's Inequality）**，它提供了对随机变量均值估计的置信界。
    *   **后悔**: UCB1在MAB问题中实现**对数后悔 $O(\log T)$**，这是理论上最优的次线性后悔。

#### 2.4.4 概率匹配 / 汤普森采样 (Probability Matching / Thompson Sampling)

*   **核心思想**: 根据对每个行动的奖励分布的**后验信念（Posterior Belief）**，按比例选择行动。
*   **方法**:
    1.  为每个行动维护一个奖励分布的先验信念（例如，对于二元奖励，使用Beta分布）。
    2.  每次选择行动前，从每个行动的**后验分布**中随机采样一个可能的平均奖励值。
    3.  选择采样值最高的行动。
    4.  观察奖励后，更新相应行动的后验分布。
*   **优点**: 经验上通常表现出色，与UCB算法相当或更好，在许多实际场景中是热门选择。它也能够实现**次线性后悔**。
*   **直观理解**: 这种方法通过“置信采样”来平衡探索与利用：它不是给不确定性大的行动一个固定的额外奖励，而是直接从不确定性中采样，让不确定性自然地引导探索。

### 2.5 总结 (Conclusion)

*   **数据高效RL的核心**: 通过优化探索与利用的权衡，以在有限数据下最小化后悔。
*   **后悔作为指标**: 提供了一个量化RL算法数据效率的正式框架。
*   **策略演进**: 从简单的贪婪方法（线性后悔）到 $\epsilon$-贪婪（固定 $\epsilon$ 时为线性后悔，衰减 $\epsilon$ 时为次线性后悔），再到 UCB 和 Thompson 采样（可实现理论最优的对数后悔），展示了RL算法在探索策略上的不断进步。
*   **MAB的意义**: MAB是理解探索与利用、后悔分析等核心概念的理想简化模型，其思想为更复杂的MDPs问题中的探索算法奠定了基础。


## Lecture 12: Bayesian RL

## 1. 关键要点 (Key Points)

-   **快速学习的目标**: 在尽量少的交互（数据）下，找到最优或接近最优的策略。这通常通过最小化**后悔（Regret）**来实现。
-   **回顾MAB中的探索策略**:
    *   **贪婪**: 线性后悔，易陷入次优。
    *   **$\epsilon$-贪婪**: 固定 $\epsilon$ 时线性后悔；衰减 $\epsilon$ 可实现次线性后悔，但需要对奖励差距有先验知识。
    *   **乐观面对不确定性（OFU）/UCB**: 通过在估计值上加上置信区间上限来指导探索，实现**次线性（对数）后悔** $O(\log T)$，无需奖励差距先验。
-   **贝叶斯老虎机（Bayesian Bandits）**:
    *   **核心**: 利用对每个行动奖励分布的**先验知识 $p[R]$**。
    *   **过程**: 维护奖励参数的后验分布，每次选择行动时，利用该后验分布来指导探索。
    *   **优势**: 如果先验知识准确，可以实现更好的性能。
-   **汤普森采样（Thompson Sampling）**:
    *   **原理**: 一种贝叶斯探索策略。在每个时间步：
        1.  根据当前后验分布，为每个行动**采样一个可能的平均奖励值**。
        2.  选择采样值最高的行动。
        3.  执行行动，观察奖励，并用贝叶斯规则**更新后验分布**。
    *   **实现概率匹配**: 汤普森采样在本质上实现了**概率匹配（Probability Matching）**，即选择行动 $a$ 的概率等于该行动是当前最优行动的后验概率。
    *   **优点**: 在实践中表现优秀，尤其在多臂老虎机和上下文老虎机中常被使用，能够实现次线性后悔。
-   **可能近似正确（Probably Approximately Correct, PAC）学习框架**:
    *   **定义**: PAC算法的目标是：在每个时间步 $t$，以**至少 $1-\delta$ 的概率**，选择一个价值**至少是 $\epsilon$-最优的行动 $Q(a) \ge Q(a^*) - \epsilon$**。
    *   **与后悔框架的区别**:
        *   后悔框架关注累积损失，目标是长期效率。
        *   PAC框架关注每次决策的质量，目标是每次决策都是“足够好”的，并在多项式时间内达成。
    *   **PAC算法**: 大多数PAC算法基于**乐观主义**或**汤普森采样**。一些PAC算法通过**乐观初始化**所有价值来鼓励探索。
-   **Gittins Index**: 在贝叶斯多臂老虎机中，Gittins Index提供了一种最优的策略，它为每个臂计算一个“真实价值指数”，并始终选择指数最大的臂。这个指数只依赖于该臂自身的统计信息和剩余时间，从而将复杂问题分解。

---

## 2. 详细解读 (Detailed Breakdown)

### 2.1 回顾与引入新的评估标准 (Recap and Introducing New Evaluation Criteria)

*   **上次回顾**: 多臂老虎机（MAB）作为数据高效RL的简化模型，并引入了“后悔”（Regret）作为评估指标，目标是实现次线性后悔。
*   **本次内容**: 贝叶斯老虎机、汤普森采样，以及可能近似正确（PAC）框架。
*   **RL算法评估的多维度**: 除了收敛性、收敛速度和经验性能，我们还需要考虑：
    *   **犯错成本**: 学习过程中犯了多少错误？
    *   **问题相关性**: 算法性能与问题特性（如臂的数量、奖励差距）的关系。

### 2.2 回顾MAB与后悔 (Recap MAB and Regret)

*   **MAB**: $(A, R)$ 元组，选择一个臂 $A_t \in A$，获得奖励 $R_t \sim R^{A_t}$。目标是最大化累积奖励。
*   **后悔 $l_t = E[V^* - Q(A_t)]$**: 单步后悔是选择的行动与最优行动之间的价值差距。
*   **总后悔 $L_T = \sum_{t=1}^T l_t$**: 累积的单步后悔。
*   **目标**: 最小化 $L_T$。理想是**次线性后悔**，如 $O(\log T)$。
*   **贪婪算法**: 线性后悔，因为可能永远锁定在次优行动上。
*   **$\epsilon$-贪婪算法**: 固定 $\epsilon$ 会导致线性后悔。只有当 $\epsilon$ 衰减到 $0$ 时才能实现次线性后悔，但这需要对奖励差距有先验知识，这在实际中往往不可得。

### 2.3 乐观面对不确定性 (Optimism Under Uncertainty, OFU)

*   **核心原则**: 选择那些“可能”具有高价值的行动。
*   **UCB1算法 (Upper Confidence Bound 1)**:
    *   选择 $A_t = \arg\max_a \left[ \hat{Q}_t(a) + \sqrt{\frac{2 \log t}{N_t(a)}} \right]$
    *   $\hat{Q}_t(a)$ 是行动 $a$ 的经验平均奖励。
    *   $\sqrt{\frac{2 \log t}{N_t(a)}}$ 是置信区间宽度，代表了行动 $a$ 价值估计的不确定性。该项：
        *   随时间 $t$ 增长而增大（鼓励总体探索）。
        *   随行动 $a$ 被选择次数 $N_t(a)$ 增加而减小（被探索越多的行动，不确定性越小，探索奖励越小）。
    *   **优点**: 能够实现**对数后悔 $O(\log T)$**，在理论上是MAB问题的最优后悔界。
    *   **乐观初始化**: 将Q值初始化为一个非常高的值，可以鼓励早期探索，因为智能体最初会乐观地认为所有行动都非常好，并逐一尝试。

### 2.4 可能近似正确 (Probably Approximately Correct, PAC) 学习框架

*   **框架目标**: PAC算法在每个时间步 $t$，以**高概率（至少 $1-\delta$）**，选择一个**足够好（$\epsilon$-optimal）**的行动。
    *   **$\epsilon$-optimal**: 价值 $Q(A_t)$ 至少为最优价值 $Q(A^*)$ 减去 $\epsilon$，即 $Q(A_t) \ge Q(A^*) - \epsilon$。
*   **关注点**: PAC框架关注算法在多项式时间内找到一个好的策略，而不是累积的后悔。
*   **PAC算法特点**: 大多数PAC算法基于乐观主义或汤普森采样。一些PAC算法通过将所有Q值初始化为一个与问题相关的**高值**来鼓励探索。

### 2.5 贝叶斯老虎机 (Bayesian Bandits)

*   **核心**: 引入对奖励分布的**先验知识 $p[R]$**。
*   **贝叶斯推断**: 在贝叶斯框架下，智能体为每个臂的未知奖励参数维护一个**后验分布**。每次观察到奖励后，使用**贝叶斯规则**更新后验分布。
    *   **伯努利奖励的共轭先验**: Beta分布是伯努利分布的共轭先验。如果奖励是0或1（伯努利），那么后验分布仍是Beta分布，参数更新非常简单。
*   **优势**: 如果先验知识准确，贝叶斯老虎机可以在学习过程中更有效地利用信息，从而实现更好的性能。

### 2.6 汤普森采样 (Thompson Sampling)

*   **原理**: 一种贝叶斯探索策略，直观且在实践中表现优秀。
    1.  **初始化**: 为每个臂的奖励分布参数设置一个先验分布（例如，Beta(1,1)代表均匀分布）。
    2.  **循环迭代**:
        *   **采样**: 对于每个臂 $a$，从其当前的后验分布中**采样一个可能的平均奖励值 $\tilde{Q}(a)$**。
        *   **选择**: 选择采样值最高的臂 $A_t = \arg\max_a \tilde{Q}(a)$。
        *   **更新**: 执行 $A_t$，观察奖励 $R_t$，并使用贝叶斯规则更新 $A_t$ 的后验分布。
*   **概率匹配**: 汤普森采样实现了一种**概率匹配**行为：选择某个行动的概率与该行动是当前最优行动的后验概率成比例。
    $$\pi(a|h_t) = P[Q(a) > Q(a'), \forall a' \ne a | h_t]$$
*   **优点**:
    *   **简单直观**: 易于理解和实现。
    *   **经验表现**: 在实践中通常与UCB算法相当或更好，甚至在上下文老虎机中表现突出。
    *   **次线性后悔**: 也能实现对数后悔 $O(\log T)$。

### 2.7 Gittins Index

*   **最优策略**: 在贝叶斯多臂老虎机中，Gittins Index提供了一种最优的策略。
*   **核心**: 它为每个臂计算一个“实时价值指数”，并始终选择当前指数最大的臂。
*   **优势**: 这个指数的计算只依赖于该臂自身的统计信息和剩余时间，从而将复杂的多臂问题分解为一系列简单的单臂决策问题。
*   **局限性**: Gittins Index的计算可能很复杂，并且仅适用于有限视野和折扣奖励。
