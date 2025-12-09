---
title: 'Attention机制发展'
date: '2025-12-9'
tags: ['深度学习', 'Attention']
---

简单写一下目前常见的一些AttentionModule，整体应该会以技术的更新路线来写。Attention的大量应用最初是Google在Transformer中运用的self-Attention和MultiHeadAttention，随后出现了各类变体，会在以下代码中写出。

## MultiHeadAttention

这是最基础的Attention，后续AttentionModule基本都在这上面做修改，将Attention操作并行运行h次，每个头通过不同的线性投影学习一组独立的QKV,最后进行拼接。h个查询头会带来h组KV矩阵，每个查询头使用其Q和对应的KV进行计算。
$$\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{Head}_1, \dots, \text{Head}_h) \mathbf{W}^O$$其中 $\text{Head}_i = \text{Attention}(Q_i, K_i, V_i)$。
本架构最大参数量为$4 \times d_{\text{model}} \times d_{\text{model}}$

```python
import math
import numpy as np
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,head_num,hidden_dim: int = 728,dropout = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.dropout = nn.Dropout(dropout)
        self.Q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,X,attention_mask = None):
        batch_size,seq_len,_ = X.size()
        Q = self.Q_linear(X)
        K = self.K_linear(X)
        V = self.V_linear(X)

        # transpose(1,2)是为了将(B,S,H,D)的向量转为(B,H,S,D)，便于GPU并行计算h个(S,D)@(D,S)乘法
        # 不转置PyTorch等框架无法直接识别(S,D)的矩阵乘法,因为H夹在两个维度间
        Q_state = Q.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        K_state = K.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        V_state = V.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)

        # 输入(B,S,H,D),转置后乘法为(B,H,S,D) @ (B,H,D,S) = (B,H,S,S)
        # 本处公式中的hidden_dim变为了head_dim
        attention_value = Q_state @ K_state.transpose(-2,-1)/math.sqrt(self.head_dim)

        # 填充掩码，读mask==0的地方进行掩码。mask的形状要求能和attention_value进行广播，通常为(B,1,1,S)或(B,1,S,S)
        if attention_mask is not None:
            attention_value = attention_value.masked_fill(
                attention_mask == 0,
                float('-1e20')
            )

        attention_weight = torch.softmax(attention_value,dim = -1)
        attention_weight = self.dropout(attention_weight)
        output = attention_weight @ V_state
        # view操作要求连续内存
        output = output.transpose(1,2).contiguous()
        # 多头拼接
        output = output.view(batch_size,seq_len,self.hidden_dim)
        output = self.output_proj(output)
        return output
```
Thought:既然目前的LLM大部分使用Decoder-only架构，用自身目前的输出作为QKV，那么attention的底层机制能否进行对这方面的优化适配？

## KV Cache

在写到后续Attention优化部分前，了解Attention机制的主要优化点是很有必要的。对于之前的方法，在我们生成到后续的token时，会重复对之前的token计算KV矩阵，而Q只计算当前token的。每次都计算KV矩阵给GPU带来了很大的开销，为了节约计算资源，KV Cache优化技术随之而生。KV Cache的思路是存储历史的KV矩阵，每次计算时无需重新计算而是使用历史矩阵。在训练阶段不需要KV Cache,只有在推理阶段才使用.

## MQA

MQA通过牺牲少量的注意力换来了速度的大幅提升，其和MHA的主要差别在于让所有的注意力头共享一套KV矩阵，理论上将KV Cache降为了原先的$1/n$. 参数量为$2 \times d_{\text{model}}^2 + 2 \times d_{\text{model}} \times d_{\text{head}}$ .其优势在于减少了参数量,激活值显存和梯度通信量.

```python
import math
import numpy as np
import torch
import torch.nn as nn


class MultiQueryAttention(nn.Module):
    def __init__(self, head_num, hidden_dim: int = 728, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.dropout = nn.Dropout(dropout)
        self.Q_linear = nn.Linear(hidden_dim, hidden_dim)
        # KV只有一个头用于计算
        self.K_linear = nn.Linear(hidden_dim, self.head_dim)
        self.V_linear = nn.Linear(hidden_dim, self.head_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        batch_size, seq_len, _ = X.size()
        Q = self.Q_linear(X)
        K = self.K_linear(X)
        V = self.V_linear(X)

        Q_state = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K_state = K.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        V_state = V.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)

        attention_value = Q_state @ K_state.transpose(-2, -1) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_value = attention_value.masked_fill(
                attention_mask == 0,
                float('-1e20')
            )

        attention_weight = torch.softmax(attention_value, dim=-1)
        attention_weight = self.dropout(attention_weight)
        output = attention_weight @ V_state
        output = output.transpose(1, 2).contiguous()
        # 多头拼接
        output = output.view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(output)
        return output
```

## GQA

GQA是MHA和MQA之间的平衡策略,把所有头分成g组,每组共享同一对KV进行计算.DeepseekV1就采用了本技术.

```python
class GroupQueryAttention(nn.Module):
    # kv_num = 1时即为MQA,kv_num = head_num时为MHA
    def __init__(self,hidden_dim,head_num,kv_num,dropout=0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.kv_num = kv_num
        self.dropout = nn.Dropout(dropout)
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)#(hidden_dim,head_dim * head_num)
        self.key_linear = nn.Linear(hidden_dim, self.kv_num*self.head_dim)
        self.value_linear = nn.Linear(hidden_dim, self.kv_num*self.head_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,X,attention_mask=None):
        batch_size,seq_len,_ = X.size()
        Q = self.query_linear(X)
        K = self.key_linear(X)
        V = self.value_linear(X)
        #(batch_size,seq_len,hidden_dim) -> (b,head_num,s,head_dim)
        q_state = Q.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        k_state = K.view(batch_size,seq_len,self.kv_num,self.head_dim).transpose(1,2)
        v_state = V.view(batch_size,seq_len,self.kv_num,self.head_dim).transpose(1,2)
        # 3*8*2*16

        kv_head = self.head_num // self.kv_num
        # 原地复制矩阵
        k_state = k_state.repeat_interleave(kv_head,dim=1)
        v_state = v_state.repeat_interleave(kv_head,dim=1)
        """k_state = k_state.unsqueeze(2).expand(-1, -1, kv_head, -1, -1)
        v_state = v_state.unsqueeze(2).expand(-1, -1, kv_head, -1, -1)

        k_state = k_state.reshape(batch_size, self.head_num, seq_len, self.head_dim)
        v_state = v_state.reshape(batch_size, self.head_num, seq_len, self.head_dim)"""

        attention_weight = q_state @ k_state.transpose(-1,-2)/math.sqrt(self.head_dim)
        # 3*8*2*16 @ 3*1*16*2

        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float('-1e20')
            )
        print(attention_weight.shape)

        attention_weight = torch.softmax(attention_weight,dim=-1)
        attention_weight = self.dropout(attention_weight)
        output_mid = attention_weight @ v_state
        output_mid = output_mid.transpose(1,2).contiguous()
        #view操作要求连续内存
        output_mid = output_mid.view(batch_size,seq_len,-1)
        #多头拼接

        output = self.output_linear(output_mid)
        return output
```

接下来到了一些相对比较新的东西,在了解新东西前,我们先简述GQA的低秩投影性质:

设输入向量 $x_i \in R^d$，分组为 $g$，每个组里有 $\frac{h}{g}$ 个头。

Query 的计算（每个头独立）：
$$ q_i^{(s,t)} = x_i W_q^{(s,t)} $$

Key 和 Value 的计算（每组共享）：
$$ k_i^{(s)} = x_i W_k^{(s)}, \quad v_i^{(s)} = x_i W_v^{(s)} $$
*(其中 $s$ 表示第几组，$t$ 表示组内的第几个头)*

$$ c_i = \left[ k_i^{(1)}, k_i^{(2)}, ..., k_i^{(g)}, v_i^{(1)}, v_i^{(2)}, ..., v_i^{(g)} \right] $$

它实际上等于输入向量乘以一个拼接后的权重矩阵：
$$ c_i = x_i \left[ W_k^{(1)}, W_k^{(2)}, ..., W_k^{(g)}, W_v^{(1)}, W_v^{(2)}, ..., W_v^{(g)} \right] $$

隐藏层维度(输入x的维度) $d$ 很大，而 $c_i$ 的维度满足：
$$ \text{dim}(c_i) = g(d_k + d_v) \ll d $$

## MLA

MLA是deepseek-V2技术报告中提出的注意力机制，其基于GQA做了一些改进。
#### MLA的初步尝试：
GQA在之前代码中repeat_interleave一步做了kv_head次复制，本质上是做了简单的线性变换，而MLA将本部分改为了更复杂的线性变换。GQA的KV基于x直接投影，复制，分割，MLA的KV都由$c_i$生成。这样好处是增加了模型的表达能力，坏处是让每组KV变的不同，反而增加了KV Cache。
#### 后续改进：
MLA结合dot-attention做了恒等变换：训练阶段照常，推理阶段使用压缩向量进行计算：
    $$ q_t^{(s)} k_i^{(s)T} = (x_t W_q^{(s)}) (c_i W_k^{(s)})^T = x_t (W_q^{(s)} W_k^{(s)T}) c_i^T $$
MLA 把 $W_q^{(s)} W_k^{(s)T}$ 合并，作为新的 Query 投影阵。
$k_i$ 可被 $c_i$ 替代。
V 的投影阵 $W_v^{(s)}$ 同理可以吸收到了输出层。
$v_i$ 也可被 $c_i$ 替代。
公式中$c_i$ 为压缩向量。这样的操作保证了可以只存$c_i$减少内存开销，将KV层的存储转到Q和output中。

补充说明：

1、 ${W}_q^{(s)}{W}_k^{(s)\top}$ 合并成一个矩阵的恒等变换，理论上只有在无限精度下才成立，实际上如果我们使用单精度尤其是BF16的话，经过变换后的精度损失往往还是挺明显的，经过多层累积后可能放大到比较可观的程度；

2、 实际上我们一般不按照 ${x}_t \left( {W}_q^{(s)}{W}_k^{(s)\top} \right)$ 来计算Q，而是按照 $\left( {x}_t {W}_q^{(s)} \right) {W}_k^{(s)\top}$ 来计算，这样虽然是串行的，但在低秩假设下计算量更少，并且理论精度的损失也更少，不过在文章中，我们仍按照 ${W}_q^{(s)}{W}_k^{(s)\top}$ 合并成一个矩阵来介绍。

然而，到目前为止，MLA有一个难以绕开的缺陷：不兼容RoPE。本部分原因我会在RoPE部分中再加探讨，在此暂时跳过。最后发布的MLA，采取了一种混合的方法——每个Attention Head的Q、K新增$d_r$个维度用来添加RoPE，其中K新增的维度每个Head共享.
________
为了降低激活值显存，MLA还对Q也做了低秩投影。这与减少KV Cache无关，主要是为了减少训练期间参数量和相应的梯度所占的显存。

**Extra**:其实在训练阶段，除了多了一步低秩投影以及只在部分维度加RoPE外，MLA与Q、K的Head Size由dk换成$d_k+d_r$的MHA基本无异。解码阶段的MLA则改为MQA形式,此时Q、K的Head Size变成了$d_c+d_r$，V的Head Size 则变成了$d_c$，按照原论文的设置，这是$d_k$、$d_v$的4倍.
那为什么还能提高推理效率呢？这又回到“瓶颈”一节所讨论的问题了，我们可以将LLM的推理分两部分：第一个Token的生成（Prefill）和后续每个Token的生成（Generation），Prefill阶段涉及到对输入所有Token的并行计算，然后把对应的KV Cache存下来，这部分对于计算、带宽和显存都是瓶颈，我们可以用MLA的MHA形式来算；但是Generation阶段由于每步只计算一个Token，实际上它更多的是带宽瓶颈和显存瓶颈，此时我们可以用MLA的MQA形式来算，从而明显提高Generation的速度。还有一个细节充分体现了这个特性。一般的LLM架构参数满足$h×dk=d$
，即num_heads * head_size = hidden_size，但DeepSeek-V2不一样，它dk=128,d=5120，但h=128，是一般设置的3倍！这是因为MLA的KV Cache大小跟h无关，增大h只会增加计算量和提升模型能力，但不会增加KV Cache，所以不会带来速度瓶颈。

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self,
                 head_num,
                 q_lora_rank,
                 qk_rope_head_dim,
                 qk_nope_head_dim,
                 kv_lora_rank,
                 v_head_dim,
                 hidden_dim: int = 728,
                 dropout = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.dropout = nn.Dropout(dropout)
        self.q_down_proj = nn.Linear(hidden_dim, q_lora_rank,bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank)
        # Q 解压: c_Q -> q_heads (包含 content 和 rope 两部分)
        self.q_up_proj = nn.Linear(q_lora_rank,
                                   head_num*(qk_nope_head_dim+qk_rope_head_dim),
                                   bias=False)
        # c_i = x * W_c
        self.kv_down_proj = nn.Linear(hidden_dim, kv_lora_rank+qk_rope_head_dim,bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)

        # k = c_i * W_k,  v = c_i * W_v ,只生成content部分
        self.kv_up_proj_k = nn.Linear(kv_lora_rank,head_num*qk_nope_head_dim,bias=False)
        self.kv_up_proj_v = nn.Linear(kv_lora_rank,head_num*v_head_dim,bias=False)
        self.output_proj = nn.Linear(head_num*v_head_dim, hidden_dim)

    def forward(self,X,attention_mask = None,past_kv_cache = None):
        batch_size,seq_len,_ = X.size()
        q_latent = self.q_down_proj(X)
        q_latent = self.q_norm(q_latent)
        q = self.q_up_proj(q_latent)
        q = q.view(batch_size,seq_len,self.head_num,
                   self.qk_nope_head_dim+self.qk_rope_head_dim)
        q_nope,q_pe = torch.split(q,[self.qk_nope_head_dim,self.qk_rope_head_dim],-1)

        kv_raw = self.kv_down_proj(X)
        kv_latent,k_pe = torch.split(kv_raw,[self.kv_lora_rank,self.qk_rope_head_dim],-1)
        c = self.kv_norm(kv_latent)

        if past_kv_cache is not None:
            past_c ,past_k_pe = past_kv_cache
            c = torch.cat([past_c, c], dim=1)
            k_pe = torch.cat([past_k_pe,k_pe],dim=1)
        current_kv_cache = (c, k_pe)
        k_nope = self.kv_up_proj_k(c)
        k_nope = k_nope.view(batch_size,-1,self.head_num,self.qk_nope_head_dim)

        v = self.kv_up_proj_v(c)
        v = v.view(batch_size,-1,self.head_num,self.v_head_dim)

        k_pe = k_pe.unsqueeze(2).expand(batch_size,-1,self.head_num,-1)
        #对q_pe和k_pe应用RoPE的代码暂时跳过

        q_final = torch.cat([q_nope, q_pe], dim=-1)
        k_final = torch.cat([k_nope, k_pe], dim=-1)

        # 转置用于 Attention: [B, H, S, D]
        q_final = q_final.transpose(1, 2)
        k_final = k_final.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q_final @ k_final.transpose(-2, -1)
        scores = scores / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)

        attn_weights = torch.softmax(scores, dim=-1)

        output = attn_weights @ v

        output = output.transpose(1, 2).contiguous().flatten(2)
        return self.o_proj(output), current_kv_cache
```

为什么kv_up_proj_k只生成 Content 部分？是因为 Position (RoPE) 部分必须“绕过”解压矩阵，才能在推理时实现“矩阵吸收”

```python
# 输出的 kv_latent_raw 包含两部分：
# Part A: c_i (用于生成 Content)
# Part B: k_pe (直接就是 Position，不需要解压)
kv_latent_raw = self.kv_down_proj(x)

# c_i 去走解压流程，k_pe 直接拿去用
kv_latent, k_pe = torch.split(kv_latent_raw, [...], dim=-1)

# 只有 c_i 被送进去了，生成 k_nope (No-RoPE / Content)
k_nope = self.kv_up_proj_k(c_i)

# Content (解压出来的) + Position (直接透传的)
k_final = torch.cat([k_nope, k_pe], dim=-1)
```
`k_pe` (RoPE部分) 是在压缩层直接生成的，没有经过 `up_proj` (解压层)。

MLA 的目标是在推理时把 $W_Q$和$W_{UK}$乘在一起吸收掉变成一个矩阵。

$$ \text{Score} = q \cdot k^T = (x W_Q) \cdot (c W_{UK})^T = x (W_Q W_{UK}^T) c^T $$

这个公式能成立的前提是：**$W_Q$ 和 $W_{UK}$ 都是固定的权重矩阵。**

RoPE 是一个旋转矩阵 $R_m$（随位置 $m$ 变化）。
如果 RoPE 包含在 $k$ 的解压过程中，公式会变成：
$$ k = (c W_{UK}) \cdot R_m $$
$$ \text{Score} = (x W_Q) \cdot ((c W_{UK}) R_m)^T $$

这时候，中间的项变成了 $W_Q \cdot R_m^T \cdot W_{UK}^T$。
$R_m$ 是随 Token 位置变化的，不能预先把它算好存下来。必须对每一个 Token、每一个位置都重新算一遍矩阵乘法。

为了既要有位置编码（RoPE），又要能矩阵吸收，DeepSeek 采取了分治法：

Content 部分 ($k_{nope}$)不含位置信息，低秩压缩，推理时这部分的 $W_{UK}$ 可以被 $W_Q$ 吸收。
Position 部分 ($k_{pe}$)：包含位置信息 (RoPE)。$x \to k_{pe}$ (直接从 down_proj 产出，或者单独一个小矩阵产出)。
推理时：这部分不能被吸收，必须显式地拼接到后面，参与 Attention 计算。但是因为 $k_{pe}$ 的维度通常很小（比如 64 维），所以这点计算开销和显存开销可以接受。

- 【【MLA】【KV Cache】10分钟了解DeepSeek-V2的创新点：MLA】 https://www.bilibili.com/video/BV1yspRzPEw8/?share_source=copy_web&vd_source=dd892536e513820814b982b74a29bd75
- https://kexue.fm/archives/10091

## NSA

NSA是Deepseek提出的一种稀疏注意力架构。相比于OpenAI提出的稀疏注意力，Deepseek的稀疏注意力每层都可学习，更为灵活。其基本分为四个阶段：
- 粗粒度分块：使用可学习的压缩函数对每个块做压缩，生成代表性key。
- 细粒度选择：用高分块内token计算分数，整块读的性能适配GPU，对连续内存访问吞吐量更高。
- 滑动窗口：只注意最近的k个token
- 门控机制：使用MLP计算门控分数，合并三个模块分数

代码：懒得写

- 【【稀疏注意力NSA】【DeepSeek】【论文精读】从NativeSparseAttention出发，到任何地方】 https://www.bilibili.com/video/BV14k4UzAEMu/?share_source=copy_web&vd_source=dd892536e513820814b982b74a29bd75

## DSA

比较重量级的东西，Deepseek-V3.2-Exp里面的新技术，在V3.2的技术报告中作为三大技术亮点之一呈现，在NSA的基础上做了进一步的改进，非常有技术美感。
- https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- https://kexue.fm/archives/6853
- 【【DSA】【深度解读】10分钟看懂最新发布的DeepSeek稀疏注意力新技术 从Sparse Attention讲起】 https://www.bilibili.com/video/BV1iynyzXEKx/?share_source=copy_web&vd_source=dd892536e513820814b982b74a29bd75

使用lightning indexer动态筛选分数，抛弃了NSA中人为制定的粗粒度，细粒度，滑动窗口的方式，直接万物皆可Transformer，让小的transformer确定大的transformer该注意什么。计算公式如下：
$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU} \left( \mathbf{q}_{t,j}^I \cdot \mathbf{k}_{s}^I \right)
$$

**公式符号说明：**

*   **$I_{t,s}$**：表示 Token $t$（查询位置）与 Token $s$（键位置）之间的**索引分数 (Index Score)**。
*   **$H^I$**：表示索引器（Indexer）的头数（Head Number）。
*   **$w_{t,j}^I$**：表示第 $j$ 个索引头的权重。
*   **$\mathbf{q}_{t,j}^I$**：索引器中第 $t$ 个位置在第 $j$ 个头的 Query 向量。
*   **$\mathbf{k}_{s}^I$**：索引器中第 $s$ 个位置的 Key 向量。

计算注意力时候，只用选择的top-k个分数进行计算。理论部分结束，比什么NSA，MLA好看的多。

这个代码实在是太长了不想写了，把Deepseek开源的代码直接贴上来吧，原网址https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py。

```python
class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
        self.softmax_scale = self.head_dim ** -0.5
        self.scale_fmt = args.scale_fmt

        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn), persistent=False)
        self.register_buffer("k_scale_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, dtype=torch.float32), persistent=False)


    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        index_score = fp8_index(q_fp8.contiguous(), weights, self.k_cache[:bsz, :end_pos].contiguous(), self.k_scale_cache[:bsz, :end_pos].contiguous())
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        topk_indices_ = topk_indices.clone()
        dist.broadcast(topk_indices_, src=0)
        assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        return topk_indices


def weight_dequant(weight, scale):
    shape = weight.shape
    assert weight.dim() == 2
    weight = weight.view(shape[0] // block_size, block_size, shape[1] // block_size, block_size).transpose(1, 2).contiguous().view(-1, block_size * block_size)
    weight = (weight.float() * scale.view(-1, 1).float()).to(torch.get_default_dtype()).view(shape[0] // block_size, shape[1] // block_size, block_size, block_size).transpose(1, 2).contiguous().view(shape)
    return weight


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.scale_fmt = args.scale_fmt
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = Indexer(args)

        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
        self.dequant_wkv_b = None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        # we use fp8 kv cache in actual deployment, so here we simulate the precision by casting kv to fp8 and then back to bf16.
        kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
        kv = (kv_fp8.view(-1, block_size).float() * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        if mask is not None:    # MHA prefill
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(kv)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
            index_mask += mask
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:                   # MQA decode
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x
```