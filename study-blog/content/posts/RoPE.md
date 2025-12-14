---
title: 'RoPE'
date: '2025-12-14'
tags: ['深度学习','Attention']
---

先简单的写一下位置编码(Positional Encoding)：在早期的word embedding模型中，词只有词义嵌入向量，而没有关于位置的信息，而词在句子中的位置又是十分重要的，因此引入了位置编码技术。早期的PE技术基本采用绝对位置编码，直接把当前的位置向量和词向量相加，典型的例子就是Attention is All You Need这篇论文中的Positional Encoding模块。绝对位置编码只自身的位置编码只于当前的位置k相关。

Transformer在这里使用的是正余弦位置编码，公式为

$$p_{k,2i} = \sin \left( \frac{k}{10000^{\frac{2i}{d}}} \right), \quad p_{k,2i+1} = \cos \left( \frac{k}{10000^{\frac{2i}{d}}} \right)$$

$p_{k,2i}$ 和 $p_{k,2i+1}$ 是位置 k 的编码向量的第 2i 和第 2i+1 个分量，d 是位置向量的维度，k 表示第几个 token （位置）


## Why we need RoPE?

正余弦位置编码采用的是加上位置向量，模型可以从attention的内积结果中分出来位置差异和语义相关度，但比较费力。RoPE采用的是乘法，相当于对向量进行旋转。这种方法把相对位置信息显式融入了注意力机制，同时有外推的潜力。
给定位置为m的token $q_m$，位置为n的token $k_n$
词嵌入向量与绝对位置信息相乘，得到 $q_m e^{im\theta}$， $k_n e^{in\theta}$
这时候我们再来计算一下两个token之间的关注度:

$$
\begin{aligned}
&<q_m e^{im\theta}, k_n e^{in\theta}> \\
&= \text{Re}[(q_m e^{im\theta}) (k_n e^{in\theta})^*] \\
&= \text{Re}[q_m k_n^* e^{i(m-n)\theta}]
\end{aligned}
$$

不得不说，这里的机制做的相当的巧妙。RoPE通过旋转矩阵的特性，使用绝对位置编码体现出了相对位置编码。RoPE在外推上具有很大的潜力。外推就是要解决训练和预测长度不一致的问题，模型在预测时可能见到训练时没有遇到的位置编码，且要处理的token更多。一个解决方法是使用掩码，只让模型看到训练长度个token。这种方法效果有限。另外的方法有：
- 线性缩放token间距离，进行外推和内插。
- NTK-aware Scaled RoPE：低频内插，高频外推，波长大于原最大序列长度时会引入未见过的旋转角度。
- NTK-by-parts：低频完全内插，高频完全外推，介于中间用NTK-aware Scaled RoPE。
- YaRN：NTK-by-parts + attention-scaling，attention-scaling就是attention score除以常数t。

对RoPE的数学机制做一些简述，首先来看它的公式：

$$(\mathbf{R}_m \mathbf{q})^T (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^T (\mathbf{R}_m^T \mathbf{R}_n) \mathbf{k} = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}$$
    1.  $(\mathbf{R}_m \mathbf{q})^T (\mathbf{R}_n \mathbf{k})$: 旋转后的Query旋转后的Key
    2.  $\mathbf{q}^T (\mathbf{R}_m^T \mathbf{R}_n) \mathbf{k}$: 核心交互
    3.  $\mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}$: 相对位置

**$\theta$: 刻画频率 (frequencies), 或者步长**
*   $\theta_i = \text{base}^{-\frac{2i}{d}} \quad (\text{base} = 10000)$
    *   $2i$: embedding dim 两两分组,
        *   $(x_{2i}, x_{2i+1}), i = [0, d/2)$,
        *   $d/2$ 个二维子空间 (subspaces)
        *   每个子空间对应一个旋转频率 $\theta$
    *   **如何理解base:** Base 定义了最慢的那个齿轮 (最高维度) 转一圈所需要的长度 (周期)。
        *   对于低维 $i=0, \theta=1, \lambda = 2\pi = 6.28$, 每隔约 6 个 Token, 这个维度的数值就会经历一轮“上-下-上”的循环。
            *   对位置变化极度敏感。$m$ 变动 1, 角度就变很大。捕捉相邻 token 的精确关系。
        *   最高维度, $i=d/2, \theta = 1/10000, \lambda = 2\pi \cdot 10000 = 62831$
            *   $m$ 变动 1, 角度几乎不变; 只有 $m$ 变动很大 (距离很远) 时, 角度才有明显区别。捕捉长距离的语义关系。
            *   **不同的维度旋转速度不同，随着维度 $i$ 增加，$\theta$ 指数级减小，旋转周期变长。**
            *   **$m\theta = m \cdot \text{base}^{-2i/d}$ : 旋转矩阵的旋转角；**
                *   固定 $i$，随着 $m$ 的增大，从上到下，我们看到了从高频震荡到低频缓动的渐变。
                *   $i = 0, \cos(m)$，每过一个 Token，它就转过很大的角度。它对位置极其敏感。
                *   $i = d/2, \cos(m/10000)$，变化极其缓慢。
*   **关于旋转**
    *   旋转必须在一个“平面”上发生，而一个平面需要 2 个维度。为了让高维向量（比如 768 维或 4096 维）能够“旋转”，RoPE 使用了一种分治策略：它将高维空间切分为许多个独立的 2D 子空间（2D Subspaces）。
*   **relative distance $(R_{\Theta,m}^d)$**
    *   $\text{Score}(m, n) = \mathbf{q}_m^T \mathbf{k}_n$
*   **token embedding seq**
    *   纵向（序列中不同的 token），每一个 token 对应一个 position id ($m$)
    *   横向（embedding 中不同的 dim），每一组 pair，对应不同的 $\theta_i$


ref:
- 【[LLM Architect] 04 RoPE 理解的几何视角，Qwen3 RoPE 计算细节，与 Interleaved 版本的对比】 https://www.bilibili.com/video/BV1cv2XBAEsA/?share_source=copy_web&vd_source=dd892536e513820814b982b74a29bd75

```python
import torch


def RoPE(X:torch.Tensor,freq:torch.Tensor,interleaved:bool = True):
    dtype = X.type
    # (Batch, Seq, Heads, HeadDim)
    shape = X.shape
    if not interleaved:
        # *shape[:-1]的意思是从开头去到最后一个元素的前一个，*是python的解包符号
        X = X.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    X = torch.view_as_complex(X.float().view(*shape[:-1], -1, 2))
    # freq 形状通常是 [seq_len, head_dim/2]
    freq = freq.view(1, X.size(1), 1, X.size(-1))

    # flatten(3) 把最后的 [head_dim/2, 2] 展平成 [head_dim]，从第3个维度开始展平
    y = torch.view_as_real(X * freq).flatten(3)

    if not interleaved:
        # 如果输入不是交错的，输出也要还原成非交错的形式
        # 将 [r0, i0, r1, i1...] 拆分并拼接回 [r0, r1..., i0, i1...]
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)

```


