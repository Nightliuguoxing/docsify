# ML - B - 浙大版 胡浩基 第三章 人工神经网络

### 人工神经网络 (神经元的数学模型)

- Artificial Neural Networks

- 人工智能 仿生学派
- 人工智能 数理学派

### MP模型

$$
y = \phi ( \displaystyle\sum_{i=1}^m \omega_i x_i + b )
$$

**向量的形式**  
$$ \omega = 
\begin{bmatrix}
{\omega_1}\\
{\omega_2}\\
{\cdots}\\
{\omega_m}\\
\end{bmatrix}
\quad \quad X = 
\begin{bmatrix}
{X_1}\\
{X_2}\\
{\cdots}\\
{X_m}\\
\end{bmatrix}
$$

$$
y = \phi ( \omega^T x + b )
$$

**设**  
神经元的输出y是输入$ X_1, \ X_2, \ \cdots, \ X_m $的函数；
$$
y = f(x_1, \ x_2, \ \cdots, \ x_m) \\
\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ≈ f(0, \ 0, \ \cdots, \ 0) + \displaystyle\sum_{i=1}^m \begin{bmatrix} \frac{\partial f}{\partial x_i} | (0, \ 0, \  \cdots, \ 0) \end{bmatrix} x_i + \cdots \\
= \displaystyle\sum_{i=1}^m \omega_i x_i + b \quad \quad
$$

### 人工神经网络 (感知器算法)

$ y = \phi ( \displaystyle\sum_{i=1}^m \omega_i x_i + b ) = \phi ( \omega^T x + b )$

**平衡关系**
$
\begin{cases}
若 \ y_i = +1, \ 则 \ w^Tx_i + b < 0 \\
若 \ y_i = -1, \ 则 \ w^Tx_i + b > 0 \\
\end{cases}
$

**感知器算法寻找$ \omega, \ b $的方法**
1. 随机选择$\omega$和$b$.
2. 取一个训练样本$(X, \ y)$
    - (i) $ 若 \ w^Tx_i + b > 0 , \ 且 \ y_i = -1 , \ 则: \ \omega = \omega-x \quad \quad b = b-1$
    - (ii) $ 若 \ w^Tx_i + b < 0 , \ 且 \ y_i = +1 , \ 则: \ \omega = \omega+x \quad \quad b = b+1$
3. 再取一个训练样本$(X, \ y)$, 回到2
4. `终止条件: `直到所有输入输出对$(X, \ y)$, 都不满足2中(i)和(ii)两个条件之一，退出循环

对于某个$X_i$, 我们定义它的增广向量如下
- 若$ y_i = +1 $, 则$ \vec x_i = \begin{bmatrix} {x_i} \\ {1} \end{bmatrix} $
- 若$ y_i = -1 $, 则$ \vec x_i = \begin{bmatrix} {-x_i} \\ {-1} \end{bmatrix} $

**感知器算法收敛定理**  
对于N个增广向量$\vec x_1, \ \vec x_2, \ \cdots, \ \vec x_N$, 如果存在一个权重向量$\omega_{opt}$, 使得对于每一个i=1~N，有$ \omega_{opt}^T \vec x_i > 0 $

