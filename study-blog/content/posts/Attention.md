---
title: 'AttentionModule'
date: '2025-07-16'
tags: ['深度学习', 'Attention']
---

### 简洁版本：

```python
import math
import numpy as np
import torch
import torch.nn as nn

class EasyAttentionModule(nn.Module):
    def __init__(self,hidden_dim: int =728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,X):
        # X.Shape is batch_size * seq_len * hidden_dim
        Q = self.query_linear(X)
        K = self.key_linear(X)
        V = self.value_linear(X)

        #attention_value = torch.matmul(Q,K.transpose(1,2)) attention可能有更高维度
        attention_value = torch.matmul(Q,K.transpose(-2,-1))
        attention_weight = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1)
        output = torch.matmul(attention_weight,V)
        return output
X = torch.randn(2,3,4)
print(X)

net = EasyAttentionModule(4)
output = net(X)
print(output)

```

### 效率优化：

```python
class AttentionModule(nn.Module):
    def __init__(self,dim: int =728) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim,dim*3)

    def forward(self,X):
        QKV = self.proj(X)
        Q,K,V = torch.split(QKV,self.dim,dim=-1)
        attention_value = torch.matmul(Q,K.transpose(-2,-1))
        attention_weight = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1)
        output = torch.matmul(attention_weight,V)
        #output = attention_weight @ V
        return output

```

### 加入细节：

```python
#启用dropout,attention_mask,output_proj
class Attention(nn.Module):
    def __init__(self, dim, dropout) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim,dim*3)
        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Linear(dim,dim)

    def forward(self,X,attention_mask=None):
        QKV = self.proj(X)
        Q,K,V = torch.split(QKV,self.dim,dim=-1)
        attention_value = Q @ K.transpose(-2,-1)
        attention_weight = attention_value / math.sqrt(self.dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float('-1e20')
            )
        print(attention_weight)
        attention_weight = torch.softmax(attention_weight,dim=-1)
        print(attention_weight)
        attention_weight = self.dropout(attention_weight)
        print(attention_weight)
        output = attention_weight @ V
        output = self.output_proj(output)
        return output

X = torch.randn(3,4,2)
print(X)
# batch_size * seq * seq
mask = torch.tensor(
    [
        [1,1,1,0],
        [1,1,0,0],
        [1,0,0,0]
     ]
)
print(mask.shape)
mask = mask.unsqueeze(1).repeat(1,4,1)
print(mask.shape)
net = Attention(2,0.01)
output = net(X,mask)
print(output)
```

### 多头注意力：

```python
class MuitiHeadAttention(nn.Module):
    def __init__(self,hidden_dim,head_num,dropout=0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.dropout = nn.Dropout(dropout)
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)#(hidden_dim,head_dim * head_num)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,X,attention_mask=None):
        batch_size,seq_len,_ = X.size()
        Q = self.query_linear(X)
        K = self.key_linear(X)
        V = self.value_linear(X)
        #(batch_size,seq_len,hidden_dim) -> (b,head_num,s,head_dim)
        q_state = Q.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        k_state = K.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        v_state = V.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)

        attention_weight = torch.matmul(q_state,k_state.transpose(-1,2))/math.sqrt(self.head_dim)

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
        output_mid = output_mid.view(batch_size,seq_len,-1)

        output = self.output_linear(output_mid)
        return output

X = torch.randn(3,2,128)
# batch_size * seq * seq
mask1 =(
    torch.tensor(
    [
        [0,1],
        [0,0],
        [1,0]
     ]
).unsqueeze(1).unsqueeze(2).expand(3,8,2,2))
print(mask1.shape)
net = MuitiHeadAttention(128,8)
output = net(X,mask1)
print(output)
```