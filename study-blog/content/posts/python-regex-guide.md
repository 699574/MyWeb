---
title: "Python正则表达式完全指南"
date: "2025-07-09"
tags: ["Python", "正则表达式"]
---

# Python正则表达式完全指南

正则表达式是处理文本的强大工具，在Python中通过re模块实现。本文将从基础到高级，全面介绍Python正则表达式的使用方法。

## 1. 基础知识

### 1.1 什么是正则表达式

正则表达式是一种用于匹配字符串中字符组合的模式。在Python中，我们使用re模块来处理正则表达式。

```python
import re
```

### 1.2 基本匹配

最简单的正则表达式就是直接匹配字符：

```python
pattern = r"hello"
text = "hello world"
result = re.search(pattern, text)
print(result.group())  # 输出: hello
```

## 2. 元字符

元字符是正则表达式中具有特殊含义的字符。

### 2.1 常用元字符

**常用元字符表**

| 元字符 | 描述 |
| ------ | ------ |
| . | 匹配除换行符外的任意字符 |
| ^ | 匹配字符串开头 |
| $ | 匹配字符串结尾 |
| * | 匹配前面的模式零次或多次 |
| + | 匹配前面的模式一次或多次 |
| ? | 匹配前面的模式零次或一次 |
| {n} | 精确匹配n次 |
| {n,} | 匹配至少n次 |
| {n,m} | 匹配n到m次 |

```python
# 匹配任意字符
re.search(r"h.llo", "hello")  # 匹配成功

# 匹配字符串开头
re.search(r"^hello", "hello world")  # 匹配成功
re.search(r"^world", "hello world")  # 匹配失败

# 匹配次数
re.search(r"a{2,4}", "aaa")  # 匹配成功，匹配"aaa"
```

### 2.2 字符类

字符类用方括号[]表示，可以匹配方括号中的任意一个字符。

```python
# 匹配a或b或c
re.search(r"[abc]", "apple")  # 匹配成功，匹配"a"

# 范围表示
re.search(r"[a-z]", "Apple")  # 匹配成功，匹配"p"

# 否定字符类
re.search(r"[^0-9]", "123a")  # 匹配成功，匹配"a"
```

### 2.3 预定义字符类

**预定义字符类表**

| 字符类 | 描述 |
| ------ | ------ |
| \d | 匹配数字，等价于[0-9] |
| \D | 匹配非数字，等价于[^0-9] |
| \w | 匹配字母、数字、下划线，等价于[a-zA-Z0-9_] |
| \W | 匹配非字母、数字、下划线，等价于[^a-zA-Z0-9_] |
| \s | 匹配空白字符（空格、制表符、换行符等） |
| \S | 匹配非空白字符 |

```python
re.search(r"\d+", "abc123")  # 匹配成功，匹配"123"
re.search(r"\w+", "hello_123")  # 匹配成功，匹配"hello_123"
```

## 3. 分组和引用

### 3.1 分组

使用圆括号()进行分组，可以提取匹配的子字符串。

```python
pattern = r"(\d{3})-(\d{4})"
text = "电话号码: 123-4567"
match = re.search(pattern, text)
print(match.group())   # 输出完整匹配: 123-4567
print(match.group(1))  # 输出第一个组: 123
print(match.group(2))  # 输出第二个组: 4567
```

### 3.2 命名分组

可以给分组命名，使引用更加清晰。

```python
pattern = r"(?P<area>\d{3})-(?P<number>\d{4})"
text = "电话号码: 123-4567"
match = re.search(pattern, text)
print(match.group("area"))    # 输出: 123
print(match.group("number"))  # 输出: 4567
```

### 3.3 反向引用

在模式中引用之前的分组。

```python
# 匹配重复的单词
pattern = r"\b(\w+)\s+\1\b"
text = "hello hello world"
match = re.search(pattern, text)
print(match.group())  # 输出: hello hello
```

## 4. 常用函数

### 4.1 re.search()

搜索整个字符串，找到第一个匹配项。

```python
result = re.search(r"\d+", "abc123def456")
print(result.group())  # 输出: 123
```

### 4.2 re.match()

从字符串开头开始匹配。

```python
# 匹配成功
result = re.match(r"\d+", "123abc")
print(result.group())  # 输出: 123

# 匹配失败
result = re.match(r"\d+", "abc123")
print(result)  # 输出: None
```

### 4.3 re.findall()

找出所有匹配项，返回列表。

```python
result = re.findall(r"\d+", "abc123def456")
print(result)  # 输出: ['123', '456']
```

### 4.4 re.finditer()

找出所有匹配项，返回迭代器。

```python
result = re.finditer(r"\d+", "abc123def456")
for match in result:
    print(match.group(), match.span())
# 输出:
# 123 (3, 6)
# 456 (9, 12)
```

### 4.5 re.sub()

替换匹配的子字符串。

```python
result = re.sub(r"\d+", "NUM", "abc123def456")
print(result)  # 输出: abcNUMdefNUM
```

## 5. 高级技巧

### 5.1 非贪婪匹配

默认情况下，*、+等是贪婪的，会尽可能多地匹配。使用?可以使它们变成非贪婪模式。

```python
# 贪婪匹配
re.search(r"a.*b", "aabab").group()  # 输出: aabab

# 非贪婪匹配
re.search(r"a.*?b", "aabab").group()  # 输出: aab
```

### 5.2 前瞻和后顾

- 正向前瞻 (?=...): 匹配一个位置，其后能匹配到指定模式
- 负向前瞻 (?!...): 匹配一个位置，其后不能匹配到指定模式
- 正向后顾 (?<=...): 匹配一个位置，其前能匹配到指定模式
- 负向后顾 (?<!...): 匹配一个位置，其前不能匹配到指定模式

```python
# 匹配后面跟着"world"的"hello"
re.search(r"hello(?=\sworld)", "hello world").group()  # 输出: hello

# 匹配不是数字的单词
re.findall(r"\b\w+\b(?!\s+\d)", "apple 1 banana 2 cherry")  # 输出: ['1', 'banana', '2']
```

### 5.3 标志(Flags)

可以使用标志来修改匹配行为：

```python
# 忽略大小写
re.search(r"python", "Python", re.IGNORECASE).group()  # 输出: Python

# 多行模式
text = "Line1\nLine2"
re.findall(r"^Line", text, re.MULTILINE)  # 输出: ['Line', 'Line']

# 点号匹配所有字符，包括换行符
re.search(r"Line.+", "Line1\nLine2", re.DOTALL).group()  # 输出: Line1\nLine2
```

## 6. 实用示例

### 6.1 提取电子邮件

```python
text = "联系方式: user@example.com 和 admin@site.org"
emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
print(emails)  # 输出: ['user@example.com', 'admin@site.org']
```

### 6.2 验证密码强度

```python
def is_strong_password(password):
    # 至少8个字符，包含大小写字母、数字和特殊字符
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    return bool(re.match(pattern, password))

print(is_strong_password("Abc123!"))  # 输出: False (长度不够)
print(is_strong_password("Abcdef123!"))  # 输出: True
```

### 6.3 解析日志文件

```python
log_line = '192.168.1.1 - - [25/Mar/2021:10:15:32 +0800] "GET /index.html HTTP/1.1" 200 1234'
pattern = r'(\d+\.\d+\.\d+\.\d+).+\[(.+?)\].+"([A-Z]+)\s+(.+?)\s+HTTP.+"\s+(\d+)\s+(\d+)'
match = re.search(pattern, log_line)
if match:
    ip, date, method, path, status, size = match.groups()
    print(f"IP: {ip}, 日期: {date}, 方法: {method}, 路径: {path}, 状态码: {status}, 大小: {size}")
```

## 7. 性能优化

### 7.1 编译正则表达式

对于重复使用的正则表达式，应该使用re.compile()进行编译，以提高性能。

```python
pattern = re.compile(r"\d+")
result1 = pattern.search("abc123")
result2 = pattern.search("def456")
```

### 7.2 避免回溯灾难

某些正则表达式可能导致回溯灾难，使匹配过程变得极其缓慢。例如：

```python
# 可能导致回溯灾难的正则表达式
pattern = r"(a+)+b"
text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaac"
```

解决方法是重写正则表达式或使用非回溯的替代方案。

## 8. 总结

正则表达式是处理文本的强大工具，掌握它可以大大提高文本处理效率。本文介绍了Python正则表达式的基础语法、常用函数和高级技巧，希望对你有所帮助。

记住，编写正则表达式时应该遵循以下原则：
- 保持简单，避免过度复杂的表达式
- 对重复使用的表达式进行编译
- 使用命名分组提高可读性
- 注意性能问题，避免回溯灾难

通过不断实践，你将能够熟练运用正则表达式解决各种文本处理问题。 