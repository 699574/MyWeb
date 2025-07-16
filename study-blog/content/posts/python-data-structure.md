---
title: 'Python数据结构入门'
date: '2025-07-09'
tags: ['Python', '数据结构']
---

## 核心数据结构概览

| 数据结构 | 可变性 (Mutability) | 有序性 (Ordering) | 元素唯一性 | 语法示例 | 核心用途 / C++/Java 类比 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| list | **可变 (Mutable)** | **有序** | 可重复 | [1, 'a', 1] | 动态数组 (std::vector, ArrayList) |
| tuple| **不可变 (Immutable)** | **有序** | 可重复 | (1, 'a', 1) | 固定大小的记录，可作为哈希键 |
| dict | **可变 (Mutable)** | **有序** (自 Python 3.7) | Key 唯一 | {'key': 'value'} | 哈希表 (std::unordered_map, HashMap) |
| set | **可变 (Mutable)** | **无序** | 元素唯一 | {1, 'a'} | 哈希集合 (std::unordered_set, HashSet) |

---

### 1. list (列表)

list 是 Python 中最常用、最灵活的序列类型。

*   **类比**: std::vector (C++) 或 ArrayList (Java)。
*   **特性**:
    *   **可变 (Mutable)**: 创建后可以随时添加、删除或修改其中的元素。
    *   **有序 (Ordered)**: 元素按插入顺序存储，每个元素都有一个确定的索引。
    *   **异构 (Heterogeneous)**: 列表中可以包含不同数据类型的元素，例如 [10, "hello", True]。

#### 语法和常用操作

```python
# 1. 创建列表
my_list = [1, 2, "python", 4.5]
empty_list = []

# 2. 访问元素 (支持负索引，-1 表示最后一个元素)
print(my_list[0])    # 输出: 1
print(my_list[-1])   # 输出: 4.5

# 3. 切片 (Slicing) - Python 特有的强大功能
# 语法: [start:stop:step] (左闭右开)
print(my_list[1:3])  # 输出: [2, "python"]

# 4. 修改元素
my_list[2] = "java"
print(my_list)       # 输出: [1, 2, 'java', 4.5]

# 5. 添加元素
my_list.append("new") # 在末尾添加
my_list.insert(1, "c++") # 在指定索引处插入
print(my_list)       # 输出: [1, 'c++', 2, 'java', 4.5, 'new']

# 6. 删除元素
my_list.pop()        # 删除并返回末尾元素
my_list.pop(1)       # 删除并返回索引为 1 的元素.后续元素向前递补
my_list.remove("java") # 删除第一个匹配的元素
print(my_list)       # 输出: [1, 2, 4.5]

# 7. 长度和成员测试
print(len(my_list))      # 输出: 3
print(2 in my_list)      # 输出: True

```
**何时使用**: 当你需要一个会动态变化的有序集合时，list 是首选。例如，存储用户输入、处理文件行等。

---

### 2. tuple (元组)

tuple 可以看作是 "不可变的列表"。

*   **类比**: 没有完美的直接类比。可以理解为 C++ 的 const std::array 或者当你想把多个值打包成一个不可变对象时。
*   **特性**:
    *   **不可变 (Immutable)**: 一旦创建，其内容就不能被修改、添加或删除。
    *   **有序 (Ordered)**: 和 list 一样，元素有固定顺序和索引。
    *   **Hashable**: 因为不可变，所以元组可以作为字典 (dict) 的键或集合 (set) 的元素，而 list 不行。

#### 语法和常用操作

```python
# 1. 创建元组
my_tuple = (1, 2, "python", 4.5)
# 注意: 单个元素的元组需要加一个逗号
single_element_tuple = (1,) 

# 2. 访问和切片 (与 list 完全相同)
print(my_tuple[0])   # 输出: 1
print(my_tuple[1:3]) # 输出: (2, 'python')

# 3. 不可变性演示
# my_tuple[0] = 100 # 这行代码会抛出 TypeError

# 4. 元组解包 (Tuple Unpacking) - 非常 Pythonic 的用法
point = (10, 20)
x, y = point
print(f"x={x}, y={y}") # 输出: x=10, y=20

```
**何时使用**:
1.  **函数返回多个值**: return x, y 实际上是返回一个元组 (x, y)。
2.  **保护数据不被修改**: 当你希望传递一组数据，并确保它不被意外改变时。
3.  **作为字典的键**: 例如，用坐标 (x, y) 作为字典的键来存储地图信息。

---

### 3. dict (字典)

dict 是 Python 的核心，是键-值 (key-value) 对的集合。

*   **类比**: std::unordered_map (C++) 或 HashMap (Java)。
*   **特性**:
    *   **可变 (Mutable)**: 可以随时增、删、改键值对。
    *   **有序 (Ordered)**: **自 Python 3.7 起，字典会保持插入顺序**。这是一个重要的现代特性，在旧版本中字典是无序的。
    *   **键必须唯一且不可变 (Hashable)**: 键不能重复，且必须是像数字、字符串、元组这样的不可变类型。值可以是任何类型。

#### 语法和常用操作

```python
# 1. 创建字典
user = {"name": "Alice", "age": 25, "city": "New York"}
empty_dict = {}

# 2. 访问值
print(user["name"]) # 输出: Alice
# 更安全的方式: .get()，如果键不存在不会报错，而是返回 None 或默认值
print(user.get("country")) # 输出: None
print(user.get("country", "USA")) # 输出: USA

# 3. 添加或修改
user["age"] = 26          # 修改现有键的值
user["email"] = "alice@example.com" # 添加新键值对
print(user) # {'name': 'Alice', 'age': 26, 'city': 'New York', 'email': 'alice@example.com'}

# 4. 删除
del user["city"]
age = user.pop("age") # 删除并返回值

# 5. 迭代 (Iteration) - 推荐的方式
# 遍历键
for key in user:
    print(key)
# 遍历值
for value in user.values():
    print(value)
# 遍历键值对 (最佳实践)
for key, value in user.items():
    print(f"{key}: {value}")

```
**何时使用**: 当你需要通过一个唯一的标识符（键）来快速查找、存储和管理关联数据时。几乎无处不在。

---

### 4. set (集合)

set 是一个无序且不包含重复元素的集合。

*   **类比**: std::unordered_set (C++) 或 HashSet (Java)。
*   **特性**:
    *   **可变 (Mutable)**: 可以添加或删除元素。
    *   **无序 (Unordered)**: 元素没有索引，你不能像 my_set[0] 这样访问。
    *   **元素唯一 (Unique)**: 自动去除重复元素。

#### 语法和常用操作

```python
# 1. 创建集合
my_set = {1, 2, 3, 2, 1} # 重复的元素会自动被忽略
print(my_set)           # 输出: {1, 2, 3}

# 从列表创建集合以去重
unique_items = set([1, 'a', 2, 'a', 3])
print(unique_items)     # 输出: {1, 2, 3, 'a'}

# 注意: 创建空集合必须用 set()，因为 {} 创建的是空字典
empty_set = set()

# 2. 添加和删除
my_set.add(4)    # 添加元素
my_set.remove(2) # 删除元素，如果不存在会报错
my_set.discard(10) # 删除元素，如果不存在也不会报错

# 3. 核心用途：数学运算
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}

print(set_a | set_b)  # 并集 (Union): {1, 2, 3, 4, 5, 6}
print(set_a & set_b)  # 交集 (Intersection): {3, 4}
print(set_a - set_b)  # 差集 (Difference): {1, 2}
print(set_a ^ set_b)  # 对称差集 (Symmetric Difference): {1, 2, 5, 6}

```
**何时使用**:
1.  **去重**: 将 list 转换为 set 是最快的去重方法。
2.  **成员测试**: element in my_set 的速度比 element in my_list 快得多（平均 O(1) vs O(n)）。
3.  **集合运算**: 需要进行交、并、差等数学运算时。

---

### 特别说明：map

在 Python 中，map **不是一个数据结构**，而是一个内置函数。这与 C++ 的 std::map 和 Java 的 Map 接口是完全不同的概念。

*   **功能**: map(function, iterable) 会将一个函数 (function) 应用到可迭代对象 (iterable，如 list) 的每一个元素上，并返回一个**迭代器 (iterator)**。
*   **惰性计算 (Lazy Evaluation)**: map 不会立即计算所有结果并存入内存，而是在你迭代它的时候才逐个计算。这在处理大数据集时非常节省内存。

#### 示例

```python
numbers = [1, 2, 3, 4]

# 使用 map 将每个数字平方
squared_iterator = map(lambda x: x * x, numbers)

# squared_iterator 此时是一个 map 对象 (迭代器)，并未计算
print(squared_iterator) # <map object at 0x...>

# 只有当你迭代它时，计算才会发生
# 通常我们会把它转换成一个 list 来查看结果
squared_list = list(squared_iterator)
print(squared_list) # 输出: [1, 4, 9, 16]

# 在现代 Python 中，列表推导式通常更受欢迎，因为它更易读
squared_list_comprehension = [x * x for x in numbers]
print(squared_list_comprehension) # 输出: [1, 4, 9, 16]
```

希望这份详细的讲解能帮助您快速掌握 Python 的核心数据结构！