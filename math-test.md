---
layout: math
title: 数学公式测试
mathjax: true
---

# 数学公式测试页面

这是一个测试页面，用来验证 MathJax 是否正常工作。

## 行内公式测试

这是一个行内公式：$E = mc^2$，它应该正确显示。

另一个例子：$\alpha + \beta = \gamma$

## 块级公式测试

这是一个块级公式：

$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

更复杂的公式：

$$
H = -J \sum_{\langle i,j \rangle} \sigma_i^z \sigma_j^z - h \sum_i \sigma_i^x
$$

## 多行公式测试

$$\begin{align}
\nabla \times \vec{E} &= -\frac{\partial \vec{B}}{\partial t} \\
\nabla \times \vec{B} &= \mu_0\vec{J} + \mu_0\epsilon_0\frac{\partial \vec{E}}{\partial t} \\
\nabla \cdot \vec{E} &= \frac{\rho}{\epsilon_0} \\
\nabla \cdot \vec{B} &= 0
\end{align}$$

## 矩阵测试

$$
A = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$

如果所有公式都能正确显示，说明 MathJax 配置成功！
