---
title: 'NumPy入门'
date: '2025-07-09'
tags: ['Python', 'NumPy']
---


### NumPy 核心ndarray对象

NumPy 的核心是ndarray (N-dimensional array），一个强大的 N 维数组对象。它是一个**同质化 (homogeneous)** 的数据容器，即所有元素必须是**相同类型**的。

#### 为什么使用 NumPy 而不是 Pythonis

| 特性 | NumPyndarray| Pythonis|
| :--- | :--- | :--- |
| **性能** | **极高**。底层由 C 语言实现，操作经过高度优化。 | **较低**。存储的是指向对象的指针，内存不连续，计算慢。 |
| **内存** | **紧凑**。元素类型相同，内存连续存储，占用空间小。 | **分散**。存储指针，额外开销大。 |
| **功能** | **向量化运算**。可以直接对整个数组进行数学运算。 | **不支持**。需要使用for循环逐个元素计算。 |
| **类型** | **同质 (Homogeneous)**。所有元素类型必须相同。 | **异构 (Heterogeneous)**。可存储不同类型的元素。 |

**向量化运算 (Vectorization) 是 NumPy 的精髓**。它允许你用简洁的语法执行批量操作，而无需编写显式的循环。这不仅代码更优雅，而且执行速度快几个数量级。

```python
import numpy as np

# Python list 方式
list1 = [1, 2, 3]
list2 = [4, 5, 6]
result_list = [x + y for x, y in zip(list1, list2)] # 需要显式循环

# NumPy ndarray 方式
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result_arr = arr1 + arr2 # 向量化加法，简洁且高效

print(result_arr) # 输出: [5 7 9]
```

---

### 1. 创建ndarray

有多种方式可以创建 NumPy 数组。

```python
import numpy as np

# 1. 从 Python 列表或元组创建
arr1d = np.array([1, 2, 3, 4])
arr2d = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3 的二维数组

# 2. 使用内置函数创建特定数组
# 创建全零数组
zeros_arr = np.zeros((2, 3)) # 参数是形状(shape)元组
# [[0. 0. 0.]
#  [0. 0. 0.]]

# 创建全一数组
ones_arr = np.ones((3, 2))
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]

# 创建单位矩阵 (Identity Matrix)
eye_arr = np.eye(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 创建等差序列 (类似 range，但支持浮点数)
arange_arr = np.arange(0, 10, 2) # start, stop, step
# [0 2 4 6 8]

# 创建指定数量的等间隔序列
linspace_arr = np.linspace(0, 1, 5) # start, stop, num_points (包含 stop)
# [0.   0.25 0.5  0.75 1.  ]

# 创建随机数数组
rand_arr = np.random.rand(2, 3) # 0-1 之间的均匀分布
randn_arr = np.random.randn(2, 3) # 标准正态分布 (高斯分布)
randint_arr = np.random.randint(0, 10, size=(2, 3)) # [0, 10) 范围内的随机整数
```

---

### 2.ndarray的重要属性

```python
arr = np.random.randint(0, 10, size=(3, 4))
# [[...],
#  [...],
#  [...]]

# 维度 (Dimensions)
print(arr.ndim) # 2

# 形状 (Shape)，返回一个元组
print(arr.shape) # (3, 4) -> 3 行 4 列

# 元素总数 (Size)
print(arr.size) # 12

# 数据类型 (Data Type)
print(arr.dtype) # int64 (在64位系统上)
```

---

### 3. 索引与切片 (Indexing and Slicing)

NumPy 的切片比 Python 列表的切片更强大，因为它能轻松处理多维数据。**NumPy 切片返回的是原始数组的视图 (View)，而不是副本 (Copy)。** 这意味着修改视图会直接影响原始数组，这是一个非常重要的特性，可以提高性能并节省内存。

```python
arr = np.arange(10) # [0 1 2 3 4 5 6 7 8 9]

# 基本切片 (与 list 类似)
slice_arr = arr[5:8] # [5 6 7]
slice_arr[1] = 999
print(arr) # 输出: [  0   1   2   3   4   5 999   7   8   9] -> 原始数组被修改！

# 如果需要副本，必须显式调用 .copy()
copy_arr = arr[5:8].copy()

# 多维数组索引
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 方式一：链式索引 (不推荐)
print(arr2d[1][2]) # 6

# 方式二：元组索引 (推荐，更高效)
print(arr2d[1, 2]) # 6

# 多维数组切片
# 获取前两行，第 1 到 2 列
#  : 表示该维度的所有元素
# 1:3 表示索引 1 到 2
sub_arr = arr2d[:2, 1:3]
# [[2 3]
#  [5 6]]
```

#### 布尔索引 (Boolean Indexing)

这是一个极其强大的功能，允许你根据条件来选择数组元素。

```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
data = np.random.randn(4, 3) # 4x3 的随机数据

# 创建一个布尔数组
is_bob = (names == 'Bob') # [ True False False  True]

# 使用布尔数组来索引 data
# 它会选择 is_bob 中为 True 对应的行
print(data[is_bob]) 
# 会返回 data 的第 0 行和第 3 行

# 也可以直接写在一起
print(data[names == 'Bob'])

# 结合条件
# 选择非 Bob 的行
print(data[~(names == 'Bob')]) # ~ 是逻辑非
# 选择 Bob 或 Will 的行
print(data[(names == 'Bob') | (names == 'Will')]) # | 是逻辑或, & 是逻辑与

# 还可以用布尔索引来赋值
data[data < 0] = 0 # 将所有负数设置为 0
```

---

### 4. 通用函数 (Universal Functions - ufuncs)

ufuncs 是对ndarray中数据执行**逐元素 (element-wise)** 操作的函数。它们是向量化操作的核心。

```python
arr = np.arange(4) # [0 1 2 3]

# 一元 ufuncs
print(np.sqrt(arr))    # [0.  1.  1.414 1.732]
print(np.exp(arr))     # [1.    2.718 7.389 20.08]

# 二元 ufuncs
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(np.add(arr1, arr2))      # [5 7 9] (等同于 arr1 + arr2)
print(np.maximum(arr1, arr2)) # [4 5 6] (逐元素比较取最大值)
```

---

### 5. 聚合函数与轴 (Aggregation and Axes)

NumPy 提供了许多聚合函数。这些函数可以作用于整个数组，也可以沿着指定的**轴 (axis)** 进行计算。

> **轴 (Axis) 的理解**:
> *   在一个 2D 数组（矩阵）中:
>     *  xis=表示沿着**行**的方向进行计算（即对每一**列**进行聚合）。
>     *  xis=表示沿着**列**的方向进行计算（即对每一**行**进行聚合）。
> *   可以想象成“压缩”指定的轴。

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# 整个数组聚合
print(arr2d.sum())   # 21
print(arr2d.mean())  # 3.5

# 沿着轴计算
# 对每一列求和 (压缩行)
print(arr2d.sum(axis=0)) # [5 7 9] (1+4, 2+5, 3+6)

# 对每一行求和 (压缩列)
print(arr2d.sum(axis=1)) # [6 15] (1+2+3, 4+5+6)
```

---

### 6. 广播 (Broadcasting)

广播是 NumPy 最强大的特性之一，它描述了 NumPy 在处理不同形状数组时如何进行算术运算。当两个数组的形状不匹配时，NumPy 会尝试**隐式地扩展 (broadcast)** 较小数组的形状，使其与较大数组的形状兼容。

**广播规则**:
1.  如果两个数组的维度数不同，将维度较少的数组的形状前面补 1，直到维度数相同。
2.  在任何一个维度上，如果一个数组的大小是 1，另一个数组的大小大于 1，那么大小为 1 的数组会被“拉伸”以匹配另一个数组的大小。
3.  如果在任何一个维度上，两个数组的大小都大于 1 且不相等，则会引发错误。

```python
# 示例 1: 数组与标量
arr = np.array([1, 2, 3])
result = arr * 2 # 标量 2 被广播到 [2, 2, 2]
# result is [2 4 6]

# 示例 2: 2D 数组与 1D 数组
arr2d = np.array([[1, 2, 3], [4, 5, 6]]) # shape (2, 3)
arr1d = np.array([10, 20, 30])           # shape (,3)

# arr1d 的 shape 变为 (1, 3)，然后被广播到 (2, 3)
# [[10, 20, 30],
#  [10, 20, 30]]
result = arr2d + arr1d
# [[11 22 33]
#  [14 25 36]]
```

广播使得向量化代码更加灵活，避免了创建不必要的中间数组，从而节省内存。


### NumPy 中执行矩阵乘积的主要方式

在 NumPy 中，有三种主要的方式来执行矩阵乘积，它们在处理不同维度数组时有不同的行为。此外，我们还会讨论一个常见的“陷阱”——* 运算符。

1.  **@ 运算符** (推荐)
2.  numpy.matmul() 函数
3.  numpy.dot() 函数
4.  * 运算符 ( **注意：这不是矩阵乘积！** )
---

### 3.@ 运算符：现代、直观的方式

自 Python 3.5 起，@ 作为中缀运算符被引入，专门用于矩阵乘法。这是目前**最推荐、最直观**的方式。

#### 特点：
-   **语义清晰**：代码 C = A @ B 明确表示你在进行矩阵乘法。
-   **行为专一**：它就是为了矩阵乘法而设计的。

#### 使用示例：

**a) 2D 数组（标准矩阵）**
对于二维数组，@ 的行为与标准的矩阵乘法完全一致。

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 维度 (2, 3)

B = np.array([[7, 8],
              [9, 10],
              [11, 12]]) # 维度 (3, 2)

# A的列数(3) == B的行数(3)，可以相乘
C = A @ B
print("A @ B =\n", C)
print("Result shape:", C.shape) # 结果维度 (2, 2)
```
输出：
```
A @ B =
 [[ 58  64]
 [139 154]]
Result shape: (2, 2)
```

**b) 高维数组（张量/批处理）**
这是 @ 和 np.matmul 的强大之处。当处理高于二维的数组时，它们会将这些数组视为一“叠”（stack）矩阵。

-   **运算规则**：它在**最后两个维度**上执行矩阵乘法，并对**前面的维度**进行广播（broadcasting）。
-   **应用场景**：这在深度学习中非常常见，比如批处理的输入数据乘以权重矩阵。

```python
# 假设我们有2个 (2x3) 的矩阵
A_stack = np.arange(12).reshape(2, 2, 3) 
# 维度 (2, 2, 3) -> 2个 (2, 3) 矩阵

# 假设我们有2个 (3x4) 的矩阵
B_stack = np.arange(24).reshape(2, 3, 4)
# 维度 (2, 3, 4) -> 2个 (3, 4) 矩阵

# 两个堆叠的矩阵相乘
# 第一个(2,3)矩阵 @ 第一个(3,4)矩阵
# 第二个(2,3)矩阵 @ 第二个(3,4)矩阵
C_stack = A_stack @ B_stack

print("A_stack shape:", A_stack.shape)
print("B_stack shape:", B_stack.shape)
print("Result shape:", C_stack.shape) # 结果维度 (2, 2, 4)
print("Result C_stack:\n", C_stack)
```
输出：
```
A_stack shape: (2, 2, 3)
B_stack shape: (2, 3, 4)
Result shape: (2, 2, 4)
Result C_stack:
 [[[ 20  23  26  29]
  [ 56  68  80  92]]

 [[224 245 266 287]
  [308 338 368 398]]]
```
在这个例子中，A_stack[0] 与 B_stack[0] 相乘，A_stack[1] 与 B_stack[1] 相乘。前面的维度 (2,) 被广播了。

---

### 4. numpy.matmul() 函数

@ 运算符实际上就是 numpy.matmul() 函数的语法糖。它们的行为**完全相同**。

```python
# 与上面的 @ 示例完全等价
C = np.matmul(A, B)
C_stack = np.matmul(A_stack, B_stack)
```

**何时使用 np.matmul() 而不是 @？**
-   当你的代码风格偏向于函数式编程时。
-   当你需要将矩阵乘法这个“操作”本身作为参数传递给另一个函数时。
-   在不支持 @ 运算符的老旧 Python 版本（< 3.5）中。

---

### 5. numpy.dot() 函数：更通用，也更复杂

np.dot() 是一个更通用的函数，它的行为取决于输入数组的维度。这使得它在某些情况下会产生与 @ 和 np.matmul 不同的结果。

**a) 两个 1D 数组（向量）**
np.dot() 计算它们的**内积（点积）**，结果是一个标量（0维数组）。

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
dot_product = np.dot(v1, v2) 
print("Dot product of v1 and v2:", dot_product) # 结果是 32
```
**注意**：np.matmul(v1, v2) 也会得到相同的结果，但 @ 在两个1D向量上直接使用在某些旧版本中可能会有问题，虽然现在通常也表现为内积。np.dot 是计算向量内积最惯用的方法。

**b) 两个 2D 数组（矩阵）**
当两个操作数都是 2D 数组时，np.dot() 的行为**与矩阵乘法 @ 完全相同**。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C_dot = np.dot(A, B)
C_at = A @ B

print("np.dot(A, B) =\n", C_dot)
print("A @ B =\n", C_at)
# 两者结果完全相同
```

**c) 数组和标量**
np.dot 支持数组与标量的乘法，等同于 *。而 @ 和 np.matmul **不支持**这种操作。

```python
A = np.array([[1, 2], [3, 4]])
scalar = 2

# print(A @ scalar) # 这会抛出 TypeError
print(np.dot(A, scalar)) # 等同于 A * 2
```

**d) 高维数组（关键区别！）**
这是 np.dot 与 np.matmul 最重要的区别。np.dot(a, b) 进行的是**张量缩并**（tensor contraction）。

-   **np.dot(a, b)**：它对 a 的**最后一个**维度和 b 的**倒数第二个**维度进行求和。
-   **np.matmul(a, b)**：它对 a 和 b 的**最后两个**维度执行矩阵乘法，并广播其他维度。

看一个例子就能明白：
```python
a = np.arange(24).reshape(2, 3, 4) # shape (2, 3, 4)
b = np.arange(24).reshape(4, 3, 2) # shape (4, 3, 2)

# a的最后一个维度(4) 和 b的倒数第二个维度(4)匹配
c_dot = np.dot(a, b)
print("Shape of a:", a.shape)
print("Shape of b:", b.shape)
print("Shape of np.dot(a, b):", c_dot.shape) # 结果 shape (2, 3, 3, 2)
```
np.dot 将 a 的前两个维度 (2, 3) 和 b 的除倒数第二个维度的其他维度 (3, 2) 组合起来，得到最终形状 (2, 3, 3, 2)。

而 np.matmul(a, b) 会因为广播失败而报错，因为它们前面的维度 (2, 3) 和 () (b只有一个维度4) 不兼容。

---

### 6. * 运算符 (numpy.multiply)：逐元素乘积

这是一个非常常见的**陷阱**，尤其是对于从 MATLAB 等语言过来的用户。在 NumPy 中，* 运算符执行的是**逐元素乘积（Hadamard Product）**，而不是矩阵乘积。

-   **要求**：两个数组的形状必须相同，或者可以通过广播机制变为相同。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 逐元素相乘
# [[1*5, 2*6],
#  [3*7, 4*8]]
element_wise = A * B 

print("A * B (Element-wise) =\n", element_wise)
```
输出：
```
A * B (Element-wise) =
 [[ 5 12]
 [21 32]]
```
这个结果与 A @ B 完全不同。

---

### 总结与最佳实践

| 操作方式            | 两个 1D 数组 (向量)        | 两个 2D 数组 (矩阵) | N-D 数组 (张量)                                      |
| ------------------- | -------------------------- | ------------------- | ---------------------------------------------------- |
| **@** (推荐)      | 内积 (返回标量)            | **矩阵乘法**        | 在最后两维上做矩阵乘法，其他维度进行广播             |
| **np.matmul()**   | 内积 (返回标量)            | **矩阵乘法**        | 在最后两维上做矩阵乘法，其他维度进行广播             |
| **np.dot()**      | 内积 (返回标量)            | **矩阵乘法**        | 对`a 的最后一个轴和 b 的倒数第二个轴进行张量缩并 |
| *             | **逐元素**乘积             | **逐元素**乘积      | **逐元素**乘积 (支持广播)                            |

**最佳实践建议：**

1.  **进行矩阵乘法**： **始终优先使用@**。它代码可读性最高，意图最明确。
2.  **计算向量内积**： 使用 np.dot(v1, v2) 是最传统、最清晰的做法。
3.  **进行逐元素乘法**： 使用 * 运算符。
4.  **避免在高维数组上使用 np.dot**：除非你非常清楚地知道你需要它独特的张量缩并行为。在大多数情况下，@ (np.matmul) 的批处理行为才是你想要的。如果需要更复杂的张量运算，可以考虑 np.einsum。
5.  **性能**：对于标准的 2D 矩阵乘法，@, np.matmul, np.dot 的底层都调用了高度优化的 BLAS/LAPACK 库，因此它们的性能几乎没有差别。选择哪个更多是基于代码的清晰度和可读性。

### 总结

*   **核心**:ndarray对象及其高效的**向量化运算**。
*   **关键特性**:
    *   **索引与切片**: 强大、灵活，尤其是**布尔索引**。注意视图 (View) 与副本 (Copy) 的区别。
    *   **通用函数 (ufuncs)**: 逐元素操作的函数库。
    *   **聚合与轴 (Axis)**: 沿着特定维度进行数据规约。
    *   **广播 (Broadcasting)**: 优雅地处理不同形状数组之间的运算。

掌握 NumPy 是进入 Python 数据科学和机器学习领域的必备前提。建议您亲手实践这些例子，感受其强大的表达能力和卓越的性能。