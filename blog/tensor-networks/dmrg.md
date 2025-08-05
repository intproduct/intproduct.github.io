---
layout: page
title: DMRG算法详解
subtitle: 密度矩阵重整化群的理论与实现
menubar: notes_menu
show_sidebar: false
toc: true
---

# DMRG算法详解

密度矩阵重整化群(Density Matrix Renormalization Group, DMRG)算法是研究一维强关联量子系统最重要的数值方法之一。

## 目录
{:.no_toc}

1. TOC
{:toc}

---

## 1. 历史背景

DMRG算法由Steven White在1992年提出，是对Wilson数值重整化群的重要改进。它解决了传统重整化群在量子系统中的困难，成为研究一维量子多体系统的标准工具。

## 2. 基本思想

### 2.1 系统-环境分解

DMRG的核心思想是将总系统分为四个部分：
- **系统块** (System Block): 左侧要增长的部分
- **环境块** (Environment Block): 右侧固定的部分  
- **单个位点**: 要添加到系统的新位点
- **超块** (Superblock): 系统+位点+环境的整体

### 2.2 密度矩阵构造

关键创新是使用密度矩阵来选择最重要的态：

$$\rho_S = \text{Tr}_E(|\psi_{ground}\rangle\langle\psi_{ground}|)$$

其中$|\psi_{ground}\rangle$是超块的基态。

## 3. 算法流程

### 3.1 无限系统算法

```python
def infinite_dmrg(H_local, max_sites, bond_dim):
    """无限系统DMRG算法"""
    
    # 初始化：单个位点作为系统和环境
    system_block = SingleSite(H_local)
    env_block = SingleSite(H_local)
    
    for i in range(max_sites // 2):
        # 构造超块哈密顿量
        H_superblock = construct_superblock_H(
            system_block, env_block, H_local
        )
        
        # 求解基态
        energy, psi_ground = eigensolver(H_superblock)
        
        # 构造约化密度矩阵
        rho = construct_density_matrix(psi_ground, system_block.dim)
        
        # 对角化密度矩阵
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        # 保留最大的bond_dim个本征值
        truncated_basis = eigenvecs[:, -bond_dim:]
        
        # 更新系统块
        system_block = update_block(system_block, truncated_basis)
        
    return energy, system_block
```

### 3.2 有限系统算法

```python
def finite_dmrg(system_blocks, env_blocks, H_local, max_sweeps):
    """有限系统DMRG扫描算法"""
    
    energy = float('inf')
    
    for sweep in range(max_sweeps):
        # 左扫描
        for i in range(len(system_blocks) - 1):
            energy = dmrg_step(
                system_blocks[i], env_blocks[-(i+1)], 
                H_local, direction='right'
            )
            
        # 右扫描  
        for i in range(len(system_blocks) - 1, 0, -1):
            energy = dmrg_step(
                system_blocks[i], env_blocks[-(i+1)],
                H_local, direction='left'
            )
            
    return energy
```

## 4. 关键技术细节

### 4.1 对称性利用

利用系统的对称性可以大大提高效率：

```python
class SymmetryBlock:
    def __init__(self, quantum_numbers, basis_states):
        self.quantum_numbers = quantum_numbers  # 如总自旋、粒子数等
        self.basis_states = basis_states
        
    def matrix_element(self, op, bra_qn, ket_qn):
        """计算算符在对称扇区间的矩阵元"""
        if not self.selection_rule(op, bra_qn, ket_qn):
            return 0.0
        # 计算非零矩阵元...
```

### 4.2 白噪声技术

为了避免数值不稳定，通常添加小的白噪声：

$$\rho \rightarrow \rho + \epsilon \cdot \mathbb{I}$$

其中$\epsilon \sim 10^{-12}$是很小的正数。

### 4.3 预测-校正方法

使用前一步的结果作为初始猜测可以加速收敛：

```python
def prediction_correction_dmrg():
    # 预测步：使用线性外推
    psi_predicted = 2 * psi_n - psi_n_minus_1
    
    # 校正步：以预测结果为初始猜测求解
    energy, psi_corrected = eigensolver(
        H_superblock, 
        initial_guess=psi_predicted
    )
    
    return energy, psi_corrected
```

## 5. 应用实例

### 5.1 海森堡模型

```python
# 一维海森堡模型的DMRG计算
def heisenberg_dmrg_example():
    # 定义局域哈密顿量
    J = 1.0  # 交换耦合常数
    
    # Pauli矩阵
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2  
    sz = np.array([[1, 0], [0, -1]]) / 2
    
    # 两位点相互作用
    H_bond = J * (np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
    
    # 运行DMRG
    energy, ground_state = infinite_dmrg(H_bond, max_sites=40, bond_dim=100)
    
    print(f"每个位点的基态能量: {energy / 40:.6f}")
    return energy, ground_state
```

### 5.2 结果分析

DMRG计算的典型结果：
- **精度**: 通常可达到$10^{-10}$的相对误差
- **效率**: 对于100个位点的系统，计算时间约几分钟到几小时
- **适用范围**: 一维系统，键维数通常在50-1000之间

## 6. 误差来源与控制

### 6.1 截断误差

主要误差来源是密度矩阵本征值的截断：

$$\epsilon_{truncation} = \sum_{i=m+1}^{d} \lambda_i$$

其中$m$是保留的态数，$d$是总态数。

### 6.2 有限尺寸效应

```python
def finite_size_scaling(sizes, energies):
    """有限尺寸标度分析"""
    # 对于临界系统: E(L) = E_∞ + a/L + b/L²
    
    def fit_func(L, E_inf, a, b):
        return E_inf + a/L + b/(L**2)
    
    popt, _ = curve_fit(fit_func, sizes, energies)
    E_infinite = popt[0]
    
    return E_infinite
```

## 7. 现代发展

### 7.1 时间演化DMRG (t-DMRG)

研究量子系统的非平衡动力学：

$$|\psi(t + dt)\rangle = e^{-iH \cdot dt} |\psi(t)\rangle$$

### 7.2 温度有限DMRG

通过虚时间演化研究有限温度性质：

$$\rho(\beta) = \frac{e^{-\beta H}}{\text{Tr}(e^{-\beta H})}$$

### 7.3 二维系统的推广

树张量网络(TTN)和投影纠缠对态(PEPS)将DMRG思想推广到高维。

## 8. 实践建议

### 8.1 参数选择

- **键维数**: 从小开始(χ=20-50)，逐步增加
- **收敛判据**: 能量变化$< 10^{-8}$
- **扫描次数**: 通常5-20次扫描足够

### 8.2 常见问题

1. **收敛慢**: 增加键维数或改善初始猜测
2. **数值不稳定**: 检查对称性实现，添加白噪声
3. **内存不足**: 使用更高效的数据结构

## 9. 小结

DMRG算法的成功证明了张量网络方法的威力：

- **理论基础**: 基于纠缠面积律的严格数学框架
- **数值效率**: 多项式复杂度vs指数级精确对角化
- **物理洞察**: 直接给出量子态的纠缠结构
- **广泛应用**: 从凝聚态到量子化学的各个领域

---

**下一篇**: [投影纠缠对态(PEPS)详解]({{ "/blog/tensor-networks/peps/" | relative_url }})

**相关笔记**: 
- [张量网络基础概念]({{ "/blog/tensor-networks/basics/" | relative_url }})
- [矩阵乘积态详解]({{ "/blog/tensor-networks/mps/" | relative_url }})
