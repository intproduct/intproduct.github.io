---
layout: post
title: 张量网络算法入门：从DMRG到PEPS
subtitle: 理解现代量子多体计算的核心工具
tags: [张量网络, DMRG, PEPS, 量子多体]
author: Qixuan Fang
series: tensor-networks-intro
---

# 张量网络算法入门

张量网络(Tensor Networks)是现代量子多体理论中最重要的数值方法之一。本文将介绍张量网络的基本概念，并重点讲解两种最重要的算法：密度矩阵重整化群(DMRG)和投影纠缠对态(PEPS)。

## 什么是张量网络？

张量网络是一种用图形化方式表示高维张量的方法。在量子多体系统中，波函数通常具有指数级的复杂度，直接存储和操作是不现实的。张量网络通过将复杂的高维张量分解为若干个低维张量的乘积，大大降低了存储和计算的复杂度。

### 基本概念

**张量(Tensor)**: 多维数组的推广，可以看作是向量和矩阵在高维空间的扩展。

**张量分解**: 将一个高维张量表示为多个低维张量的乘积形式。

**纠缠熵**: 衡量量子系统中子系统间纠缠程度的物理量。

## DMRG算法

密度矩阵重整化群(Density Matrix Renormalization Group, DMRG)是Steven White在1992年提出的算法，用于研究一维强关联电子系统。

### 核心思想

DMRG的核心思想是通过保留系统最重要的量子态来实现降维：

1. **系统分割**: 将系统分为左右两部分
2. **密度矩阵构造**: 构造左(右)子系统的约化密度矩阵
3. **本征值分解**: 保留密度矩阵最大的几个本征值对应的本征态
4. **迭代优化**: 逐步扩大系统并重复上述过程

### 算法步骤

```python
# DMRG算法伪代码
def dmrg_algorithm(hamiltonian, bond_dim=100):
    # 初始化
    psi = random_mps(N, bond_dim)
    
    for sweep in range(max_sweeps):
        # 左扫描
        for i in range(N-1):
            # 构造局域有效哈密顿量
            H_eff = construct_effective_hamiltonian(i)
            # 求解本征值问题
            energy, psi_ground = eigensolver(H_eff)
            # 更新张量
            update_tensor(psi, i, psi_ground)
            
        # 右扫描
        for i in range(N-1, 0, -1):
            # 类似的操作...
            
    return energy, psi
```

## PEPS算法

投影纠缠对态(Projected Entangled Pair States, PEPS)是DMRG在二维系统的推广，由Verstraete和Cirac在2004年提出。

### 二维张量网络的挑战

在二维系统中，纠缠结构比一维复杂得多：
- **面积律**: 二维系统的纠缠熵与边界成正比
- **计算复杂度**: 精确收缩二维张量网络是NP-hard问题
- **近似方法**: 需要发展高效的近似算法

### PEPS的优势

1. **自然的二维结构**: 直接适用于二维晶格系统
2. **灵活的边界条件**: 可以处理开边界和周期边界
3. **可扩展性**: 原则上可以处理任意大小的系统

## 应用实例

### 1. 海森堡模型

```python
# 一维海森堡模型的DMRG计算
import numpy as np
from tensors import DMRG

# 定义海森堡哈密顿量
def heisenberg_hamiltonian(J=1.0, N=20):
    H = 0
    for i in range(N-1):
        H += J * (sigma_x[i] @ sigma_x[i+1] + 
                  sigma_y[i] @ sigma_y[i+1] + 
                  sigma_z[i] @ sigma_z[i+1])
    return H

# DMRG计算
dmrg = DMRG(bond_dimension=100)
energy, state = dmrg.solve(heisenberg_hamiltonian())
print(f"基态能量: {energy:.6f}")
```

### 2. 横场伊辛模型

二维横场伊辛模型是测试PEPS算法的经典例子：

$$H = -J \sum_{\langle i,j \rangle} \sigma_i^z \sigma_j^z - h \sum_i \sigma_i^x$$

这个模型在临界点附近展现丰富的量子相变现象。

## 前沿发展

### 1. 无限系统DMRG (iDMRG)
- 处理无限大系统
- 研究相变和临界现象

### 2. 时间演化算法 (TEBD/TDVP)
- 研究量子系统的非平衡动力学
- 量子淬火和周期驱动系统

### 3. 机器学习增强
- 结合神经网络的张量网络
- 自动优化的张量分解

## 小结

张量网络算法为我们提供了研究强关联量子系统的强大工具。从一维的DMRG到二维的PEPS，这些方法在凝聚态物理、量子化学和量子信息等领域都有重要应用。

在下一篇文章中，我们将深入探讨张量网络的数学基础和实现细节。

---

**参考文献**:
1. S. R. White, Phys. Rev. Lett. 69, 2863 (1992)
2. F. Verstraete and J. I. Cirac, arXiv:cond-mat/0407066 (2004)
3. U. Schollwöck, Annals of Physics 326, 96 (2011)
