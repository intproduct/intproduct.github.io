---
layout: page
title: 张量网络基础概念
subtitle: 理解量子多体计算的数学框架
menubar: notes_menu
show_sidebar: false
toc: true
---

# 张量网络基础概念

张量网络是现代量子多体理论中最重要的数学工具之一。本文将从基础概念开始，系统介绍张量网络的数学结构和物理意义。

## 目录
{:.no_toc}

1. TOC
{:toc}

---

## 1. 什么是张量？

### 1.1 数学定义

**张量(Tensor)**是向量和矩阵在高维空间的推广。数学上，一个$n$阶张量是一个具有$n$个指标的多维数组：

$$T_{i_1, i_2, \ldots, i_n}$$

其中每个指标$i_k$可以取$1, 2, \ldots, d_k$的值。

### 1.2 物理意义

在量子多体系统中，张量通常表示：
- **波函数系数**: $\psi_{i_1, i_2, \ldots, i_N}$
- **算符矩阵元**: $\langle i|H|j \rangle$  
- **纠缠态的表示**: Schmidt分解系数

### 1.3 张量的图形表示

我们用图形符号来表示张量：

```
    |
    |
----●----  ← 三阶张量
    |
    |
```

- 圆点代表张量
- 线段代表张量的指标
- 连接的线表示指标收缩

## 2. 张量收缩

### 2.1 基本概念

**张量收缩**是张量网络中的基本操作，相当于矩阵乘法的推广：

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

图形表示：
```
i ----●---- k ----●---- j  =  i ----●---- j
      A           B               C
```

### 2.2 Einstein求和约定

在张量记号中，重复指标表示求和：
$$C_{ij} = A_{ik} B_{kj}$$

这种约定大大简化了张量运算的表示。

### 2.3 复杂收缩示例

考虑更复杂的张量网络：

```python
# Python代码示例
import numpy as np

# 定义三个张量
A = np.random.rand(2, 3, 4)  # 形状: (i, j, k)
B = np.random.rand(4, 5, 6)  # 形状: (k, l, m) 
C = np.random.rand(6, 2)     # 形状: (m, i)

# 张量收缩: C[i,l] = A[i,j,k] * B[k,l,m] * C[m,i]
result = np.einsum('ijk,klm,mi->jl', A, B, C)
```

## 3. 量子态的张量分解

### 3.1 简单分离态

考虑两个量子比特的分离态：
$$|\psi\rangle = |\phi_1\rangle \otimes |\phi_2\rangle$$

可以写成张量形式：
$$\psi_{i_1 i_2} = \phi_1^{i_1} \phi_2^{i_2}$$

### 3.2 Schmidt分解

对于一般的二分量子态，可以进行Schmidt分解：
$$|\psi\rangle = \sum_{\alpha=1}^{\chi} \lambda_\alpha |\phi_\alpha^L\rangle \otimes |\phi_\alpha^R\rangle$$

其中$\lambda_\alpha$是Schmidt系数，$\chi$是Schmidt秩。

### 3.3 矩阵乘积态 (MPS)

对于一维量子系统，可以将波函数表示为矩阵乘积态：

$$|\psi\rangle = \sum_{\{i\}} \text{Tr}[A^{i_1} A^{i_2} \cdots A^{i_N}] |i_1 i_2 \cdots i_N\rangle$$

图形表示：
```
|i₁⟩  |i₂⟩  |i₃⟩      |iₙ⟩
  |     |     |          |
--●-----●-----●-- ... --●--
 A¹    A²    A³        Aᴺ
```

## 4. 纠缠熵与面积律

### 4.1 纠缠熵的定义

对于双分系统，纠缠熵定义为：
$$S = -\text{Tr}(\rho_A \log \rho_A)$$

其中$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$是约化密度矩阵。

### 4.2 面积律

许多物理系统的基态满足**面积律**：
- 一维系统: $S \sim \log L$ (对数修正)
- 二维系统: $S \sim L$ (与边界长度成正比)
- 三维系统: $S \sim L^2$ (与边界面积成正比)

### 4.3 张量网络的表示能力

面积律是张量网络有效性的理论基础：
- MPS可以高效表示一维基态
- PEPS适用于二维系统
- 更高维度需要更复杂的张量网络结构

## 5. 常见张量网络结构

### 5.1 矩阵乘积态 (MPS)
- **适用系统**: 一维量子链
- **键维数**: $\chi$
- **存储复杂度**: $O(N \chi^2 d)$

### 5.2 投影纠缠对态 (PEPS)  
- **适用系统**: 二维量子格子
- **键维数**: $D$
- **存储复杂度**: $O(N D^4 d)$

### 5.3 多尺度纠缠重整化ansatz (MERA)
- **特点**: 分层结构，适合临界系统
- **应用**: 共形场论，临界现象

### 5.4 树张量网络 (TTN)
- **结构**: 树状连接
- **优势**: 收缩复杂度较低
- **应用**: 分子系统，量子化学

## 6. 数值实现要点

### 6.1 键维数截断

实际计算中需要截断键维数以控制计算复杂度：

```python
def truncate_svd(matrix, max_bond_dim, cutoff=1e-12):
    """SVD截断保持最重要的奇异值"""
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # 根据截断误差和最大键维数确定保留的奇异值数量
    keep = min(max_bond_dim, np.sum(s > cutoff))
    
    return U[:, :keep], s[:keep], Vh[:keep, :]
```

### 6.2 正交化

保持张量网络的正交性有助于数值稳定性：

```python
def left_orthogonalize(tensor):
    """左正交化张量"""
    shape = tensor.shape
    matrix = tensor.reshape(shape[0], -1)
    Q, R = np.linalg.qr(matrix)
    
    new_tensor = Q.reshape(shape[0], Q.shape[1])
    return new_tensor, R
```

## 7. 小结

张量网络为我们提供了理解和计算量子多体系统的强大框架。关键要点包括：

1. **数学基础**: 张量收缩和图形表示
2. **物理意义**: 量子纠缠的高效参数化
3. **计算优势**: 避免指数级的存储需求
4. **适用范围**: 从一维到高维的各种量子系统

在下一节中，我们将具体介绍矩阵乘积态(MPS)的详细构造和操作。

---

## 参考资料

1. **书籍**:
   - "A Practical Introduction to Tensor Networks" - Roman Orus
   - "Tensor Network Theory" - Jacob Biamonte

2. **综述文章**:
   - J. Eisert et al., Rev. Mod. Phys. 82, 277 (2010)
   - R. Orús, Annals of Physics 349, 117 (2014)

3. **开源代码**:
   - [TensorNetwork](https://github.com/google/TensorNetwork)
   - [ITensor](https://itensor.org/)

---

**下一篇**: [矩阵乘积态详解]({{ "/blog/tensor-networks/mps/" | relative_url }})
