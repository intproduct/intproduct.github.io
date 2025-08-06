---
layout: page
title: TDVP算法
subtitle: 含时变分原理算法
menubar: notes_menu
show_sidebar: false
toc: true
---

# TDVP与tanTRG

## MPS流形

**参考文献：**
- PRL. 107. 070601
- PRB. 94. 165116

### 参数空间

局域张量 $ A^i \in \mathbb{C}^{D_{i-1} \times d_i \times D_i} $

这样的MPS：
$$
\mathcal{T} = \mathrm{Tr} \prod_i A^i
$$
可看作映射：
$$
\Psi: \mathcal{A} \to \mathcal{H}, \quad \mathcal{A} = \bigoplus_{i=1}^N \mathbb{C}^{D_{i-1} \times d_i \times D_i}
$$
于是：
$$
A = \{A_i\} \mapsto |\Psi(A)\rangle
$$

### MPS的规范变换

在局域张量之间可以插入单位矩阵：
$$
I_i = g_i g_i^T
$$
因此规范变换的矩阵属于一般线性群 $ g_i \in GL(D_i, \mathbb{C}) $，整体规范群为 $ G = \prod_i GL(D_i, \mathbb{C}) $。

我们有如下结论：
$$
|\Psi(A)\rangle = |\Psi(B)\rangle \iff \exists g \in G, \, B = g \cdot A
$$

### MPS流形

**参考文献：** J. Math. Phys. 55, 021902

给定 $\{D_i\}$，对所有满秩（full-rank）的 $A_i$，MPS流形是希尔伯特空间 $\mathcal{H}$ 的一个嵌入子流形，定义为商空间：
$$
\mathcal{M} = \mathcal{A} / G
$$

## 切空间近似

### 几何观点下的薛定谔方程

薛定谔方程：
$$
\frac{d}{dt} |\Psi\rangle = -i H |\Psi\rangle
$$
由于存在同构：
$$
\mathcal{H} \simeq T_{|\Psi\rangle}(\mathcal{H})
$$
该方程可视作 $\mathcal{H}$ 上的一个切向量场。

由此可引入 TDVP（Time-Dependent Variational Principle）方程：
$$
\Psi_*: \frac{dA}{dt} \mapsto \frac{d}{dt} |\Psi(A)\rangle
$$

但问题在于：参数空间的切向量是否能保证映射到$\mathcal{H}$后仍落在切空间$T_{|\Psi\rangle}\mathcal{M}$上？**不能**，因为受到 MPS ansatz 的限制。

TDVP 方程定义为：
$$
\frac{d}{dt} |\Psi\rangle = \mathbf{P}[-i H |\Psi\rangle]
$$
其中 $\mathbf{P}$ 是投影算子，确保演化方向始终位于 $T_{|\Psi\rangle}\mathcal{M}$ 内。

> **TDVP 物理意义**：在给定变分 ansatz 流形下，实现无穷小时间步长内的最优量子态演化。

目标：找到参数 $\{A_i\}$ 的演化方程。

## MPS参数的TDVP方程

一般的切向量形式为：
$$
\frac{d}{dt} |\Psi(A)\rangle = \mathrm{Tr} \sum_i \left( \prod_j A_j \right) \frac{dA_i}{dt} = \mathrm{Tr} \sum_i \left( \prod_j A_j \right) B_i
$$
我们希望这是由切映射诱导的：
$$
\Psi_*: \frac{dA}{dt} \mapsto \frac{d}{dt} |\Psi(A)\rangle
$$

规范自由度 $g_i$ 的存在意味着需进行规范固定。规范固定后，有效自由度为每格点 $D^2 d - D^2$。

采用**左正交规范**（简化内积计算，使度规为欧式）：

内积定义为：
$$
\langle \Psi_A(B), \Psi_A(B') \rangle = \sum_i \langle B_i, B'_i \rangle
$$

于是问题转化为约束优化问题：对任意切向量 $X_A \in \mathcal{H}$，在左正交条件下求解：
$$
\min_B \| \Psi_A(B) - X_A \|^2
$$

范数平方展开为：
$$
\| \Psi_A(B) - X_A \|^2 = \sum_i \langle B_i, B_i \rangle - \langle B_i, X_i \rangle - \langle X_i, B_i \rangle
$$

引入拉格朗日乘子 $\boldsymbol{\lambda}$，构造泛函：
$$
L = \sum_i \langle B_i, B_i \rangle - \langle B_i, X_i \rangle - \langle X_i, B_i \rangle - \boldsymbol{\lambda} \langle A_i, B_i \rangle
$$

对 $B_i$ 共轭求导，并与左正交条件结合（$\langle A_i, B_i \rangle = 0$），得：
$$
2\langle B_i, A_i \rangle - \langle X_i, A_i \rangle - \boldsymbol{\lambda} \langle A_i, A_i \rangle = 0
\Rightarrow \boldsymbol{\lambda} = -\langle X_i, A_i \rangle
$$

最终得到 TDVP 参数演化方程：
$$
i \frac{d}{dt} A_i = H_i A_i
$$
其中 $H_i$ 为有效哈密顿量作用于局域张量的结果。

## TDVP sweep

### Lie-Trotter分解

考虑线性演化方程：
$$
i \frac{d}{dt} A_i = -i H_i^{(1)} A_i, \quad \frac{d}{dt} s_i = i H_i^{(0)} s_i
$$
其形式解为：
$$
A_i(t) = e^{-i H_i^{(1)} t} A_i(0), \quad s_i(t) = e^{i H_i^{(0)} t} s_i(0)
$$
前者称为**正向演化**，后者称为**反向演化**。

完整演化过程如下：
- 正向：单格点依次演化至 bond，
- bond 分解后与下一格点合并，进行反向演化。

若记此过程为 $\phi$，则满足：
$$
\phi_L^*(\tau) = \phi_L^{-1}(-\tau) = \phi_R(\tau)
$$
该性质使得误差从 $O(\tau)$ 提升至 $O(\tau^2)$。

整个 sweep 包含：
- $N$ 次单点正向演化，
- $N-1$ 次反向演化。

### Lanczos方法

与 1D DMRG 类似，区别在于：
- DMRG 直接对有效哈密顿量进行对角化；
- 此处将 $e^{-i H^{(1)} \tau}$ 展开至一阶，用 Lanczos 方法求解虚时演化。

收缩复杂度一致，约为：
$$
O(D^3 w d) + O(D^2 d^2 w^2) + O(D^3 w d)
$$
其中 $w$ 为哈密顿量 MPO 的宽度。

### Auto-Differentiation方法

此方法可能完全绕开传统 DMRG 流程，直接通过自动微分优化能量或作用量，适用于现代机器学习框架集成。

### 两点TDVP

类似于两点 DMRG，在正向演化时引入 SVD。每次先将两个相邻张量合并，再演化并分解。

流程包含：
- $N-1$ 次两点正向演化，
- $N-2$ 次单点反向演化。

相比单点 TDVP，两点方法多出物理指标带来的额外计算开销，但精度更高。

### Controlled Bond Expansion (CBE)

**参考文献：**
- PRL. 130. 246402
- PRL. 133. 026401

该方法动态控制 bond 维度增长，避免过早截断，在低温或强关联区域保持精度的同时控制计算成本。

## tanTRG方法

### 变分流形上的密度矩阵

设 $\rho$ 是某变分流形上的密度矩阵，最小化自由能泛函：
$$
F[\rho] = \mathrm{Tr}(\rho H) + \frac{1}{\beta} \mathrm{Tr}(\rho \ln \rho)
$$
在全空间 $\mathcal{H}$ 中，极小化解为：
$$
\min_\rho F[\rho], \quad \mathrm{Tr} \rho = 1
\Rightarrow \rho = \frac{e^{-\beta H}}{\mathrm{Tr}(e^{-\beta H})}
$$

**困难**：MPO 形式的 $\ln \rho$ 难以直接计算。

**解决方法**：采用虚时演化方程：
$$
\begin{cases}
\frac{d\rho}{d\beta} = -\mathbf{P}\left( (H - \langle H \rangle) \rho \right) \\
\rho(0) = \mathbb{I}_{\mathcal{H}}, \quad D=1
\end{cases}
$$
注意：虚时演化本身非严格变分，但在 $D \to \infty$ 极限下，TDVP 可精确还原虚时演化（类似 DMRG 行为）。

### MPO的TDVP

MPO 与 MPS 结构相似，可视为维度为 $d^2$ 的 MPS —— 将每个局域矩阵视为超矢量：
$$
\rho \mapsto |\rho\rangle
$$
因此 TDVP 方法可直接推广至 MPO。

**技术细节**：
- 在构建有效哈密顿量时，上方环境的一个 MPS 替换为 MPO，导致多出一个指标；
- 需进行一次内部收缩以消除多余指标，其余流程与 MPS-TDVP 一致。

**其他技术要点**：

#### SETTN 初始化
$$
\rho(\tau \ll 1) = e^{-\beta H} \approx 1 - \tau H + \frac{1}{2} \tau^2 H^2 + \cdots
$$
在高温极限下展开，构造二阶 MPO。可通过变分乘法（variational multiplication）和变分求和（variational sum）高效实现。

**参考文献：**
- PRB. 95. 161104

> 初始态为无限高温态（$\beta=0$），此时 $\rho = \mathbb{I}/\dim(\mathcal{H})$，对应 $D=1$。但此时 TDVP 投影误差较大（依赖于 $D$），故需用展开式初始化以提高稳定性。

#### 温度步长的选择

温度需离散采样，选择策略如下：

经验稳定方法：
- 在 $\beta \sim 1$ 附近前段采用**指数增长步长**：$\Delta \beta = 2^n$
- 后段趋于平衡区后改为**线性增长**：每次增加固定量（如 $\Delta \beta = 1$）

能量单位设为 $t=1, J=1$。

## 自旋动力学计算

动力学结构因子定义为：
$$
S^{\alpha\beta}(k,\omega) = \frac{1}{N} \sum_{ij} e^{-i k (r_i - r_j)} \int dt\, e^{i \omega t} \langle e^{i H t} S_i^\alpha e^{-i H t} S_j^\beta \rangle
$$

主要任务是计算期望值 $\langle \cdots \rangle$。若时间演化作用于基态 $|\psi_g\rangle$，可提出相位因子：
$$
e^{i E_g t} \langle \psi_g | S_i^\alpha e^{-i H t} S_j^\beta | \psi \rangle
$$

注意到算符分别作用于初态和末态，即：
- $S_j^\beta$ 作用于基态产生激发态；
- $e^{-i H t}$ 进一步演化；
- $S_i^\alpha$ 与之做内积。

因此可通过 TDVP 方法演化含时态 $e^{-i H t} S_j^\beta |\psi_g\rangle$，再与 $S_i^\alpha |\psi_g\rangle$ 收缩计算内积，全部过程转化为张量网络的高效收缩。