# ML - B1 - 浙大版 胡浩基 第二章 支持向量机

### 支持向量机 (线性可分定义) 

- 线性可分 (Linear Separable)
- 线性不可分 (Nolinear Separable)

###### 用`数学`雅严格定义训练样本以及他们的标签  
`假设`: 我们有N个训练样本和他们的标签  
$
\{(X_1, \ y_1), \ (X_2, \ y_2), \ \cdots, \ (X_N, \ y_N)\} \\
其中: X_i = [x_{i1}, \ x_{i2}]^T \\
\quad \quad \quad \ y_i = \{-1, \ 1 \}
$

###### 用`数学`严格的定义线性可分
线性可分的严格定义：一个训练样本集$\{(X_i, \ y_i), \ \cdots, \ (X_N, \ y_N)\}$, 在i = 1\~N 线性可分，是指存在$(\omega_1, \ \omega_2, \ b)$, 使得对i = 1\~N， 有：
1. 若$y_i = +1，\ 则 \ \omega_1 x_{i1} + \omega_2 x_{i2} + b > 0$
2. 若$y_i = -1，\ 则 \ \omega_1 x_{i1} + \omega_2 x_{i2} + b < 0$

###### 用`向量形式`来定义线性可分
`假设`：
$ X_i = 
\begin{bmatrix}
{x_{i1}}\\
{x_{i2}}\\
\end{bmatrix}^T
$
$
\omega = 
\begin{bmatrix}
{\omega_1}\\
{\omega_2}\\
\end{bmatrix}^T
$

1. 若$y_i = +1，\ 则 \ \omega^T X_{i} + b > 0$
2. 若$y_i = -1，\ 则 \ \omega^T X_{i} + b < 0$

###### 线性可分定义的`最简化形式`
1. 若$y_i = +1，\ 则 \ \omega^T X_{i} + b > 0$
2. 若$y_i = -1，\ 则 \ \omega^T X_{i} + b < 0$

`如果`：
$y_i = +1 \ 或 \ -1$  
一个训练样本集$\{(X_i, \ y_i) \}$,在i = 1\~N 线性可分，是指存在$(\omega, \ b)$，使得对i = 1\~N, 有 $y_i(\omega^T X_i + b) > 0$

### 支持向量机 (问题描述) 

支持向量机寻找最优分类直线应满足：
1. 该直线分开了两类；
2. 该直线最大化`间隔`(Margin)；
3. 该直线处于间隔的中间，到所有支持向量距离相等。

### 支持向量机 (优化问题) 

假定训练样本集是线性可分的，支持向量机需要寻找的是最大化`间隔`的超平面：  
最小化(Minimize)：$\frac{1}{2}{\Vert\omega\Vert}^2$  
限制条件：$y_i(\omega^T x_i + b) \ge 1, \quad (i=1, \ \cdots, \ N)$

- `已知`：训练样本集$\{ (X_i, \ y_i) \}, \quad (i=1, \ \cdots, \ N)$;
- `待求`：$(\omega, \ b)$

###### 二次规划的定义:
1. `目标函数`是二次项。
2. `限制条件`是一次项。
- 要么`无解`，要么只有`唯一`的最小值

### 支持向量机 (线性不可分情况)
对于线性不可分情况，需适当放松限制条件。
限制条件改写：$y_i(\omega^T X_i + b) \ge 1 - \delta_i (i=1, \ \cdots, \ n)$   
改造后的支持向量机优化版本  
最小化(Minimize)：$\frac{1}{2}{\Vert\omega\Vert}^2 + C\displaystyle\sum_{i=1}^N \delta_i \ 或 \ \frac{1}{2}{\Vert\omega\Vert}^2 + C\displaystyle\sum_{i=1}^N \delta_i^2 \quad $ C为超参数由人工赋值  
限制条件：$(1) \quad \delta_i \ge 0, \quad (i=1, \ \cdots, \ N)$  
$\quad \quad \quad \ \ \ \  (2) \quad y_i(\omega^T x_i + b) \ge 1 - \delta_i, \quad (i=1, \ \cdots, \ N)$

### 支持向量机 (低维到高维的映射)
二维到五维的映射$\phi(x)$使线性不可分变成了线性可分

假设：  
在一个`M维空间`上随机取N个训练样本随机的对每个训练样本赋予标签+1或-1  
`假设`：  
这些训练样本线性可分的概率为`P(M)`  
当M趋于无穷大时 P(M) = 1
再次优化：  
最小化(Minimize)：$\frac{1}{2}{\Vert\omega\Vert}^2 + C\displaystyle\sum_{i=1}^N \delta_i \ 或 \ \frac{1}{2}{\Vert\omega\Vert}^2 + C\displaystyle\sum_{i=1}^N \delta_i^2 \quad $ C为超参数，由人工赋值  
限制条件：$(1) \quad \delta_i \ge 0, \quad (i=1, \ \cdots, \ N)$  
$\quad \quad \quad \ \ \ \  (2) \quad y_i[\omega^T\phi(X_i) + b] \ge 1 - \delta_i, \quad (i=1, \ \cdots, \ N)$