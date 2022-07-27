# ML - B1 - 浙大版 胡浩基 第二章 支持向量机

### 支持向量机 (线性可分定义) 

- 线性可分 (Linear Separable)
- 线性不可分 (Nolinear Separable)

###### 用`数学`严格定义训练样本以及他们的标签  
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

### 支持向量机 (核函数的定义)

核函数 ( Kernel Function ) : $K(X_1, \ X_2) = \phi(X_1)^T \phi(X_2)$

### 支持向量机优化问题
$K(X_1, \ X_2)$能写成$\phi(X_1)^T \phi(X_2)$的充要条件
1. $K(X_1, \ X_2) = K(X_2, \ X_1)$ (交换性)
2. $ \forall C_i(1 = 1, \ \cdots, \ N) $, $ \forall N $ 有 $ \displaystyle\sum_{i=1}^N \displaystyle\sum_{j=1}^N C_i C_j K (X_i X_j) \ge 0 $ (半正定性)

### 支持向量机 (原问题和对偶问题)

> 原问题  

最小化 : $f(\omega)$  
限制条件 : $g_i(\omega) \le 0 \quad (i = 1, \ \cdots, \ K)$  
$\quad \quad \quad \ \ \  h_i(\omega) = 0 \quad (i = 1, \ \cdots, \ m)$

> 对偶问题

$$
L(\omega, \ \alpha, \ \beta) \ = \ f(\omega) + \displaystyle\sum_{i=1}^K \alpha_i g_i (\omega) + \displaystyle\sum_{i=1}^K \beta_i h_i (\omega) \\
\quad \quad = \ f(\omega) + \alpha^T g(\omega) + \beta^T h(\omega) 
$$
$
其中 \ \alpha \ = \ [\alpha_1, \ \alpha_2, \ \cdots, \ \alpha_K]^T \\
\quad \quad \ \beta \ = \ [\beta_1, \ \beta_2, \ \cdots, \ \beta_M]^T \\
\quad \quad \ g(\omega) = [g_1(\omega), \ g_2(\omega), \ \cdots, \ g_k(\omega), \ ]^T \\
\quad \quad \ h(\omega) = [h_1(\omega), \ h_2(\omega), \ \cdots, \ h_M(\omega), \ ]^T \\
$
最大化 : $\theta(\alpha, \ \beta) = inf \ L(\omega, \ \alpha, \ \beta)$, 所有定义域内的 $ \omega $  
限制条件 : $\alpha_i \ge 0, \quad (i = 1, \ \cdots, \ K)$

综合`原问题`和`对偶问题`的定义可以得到：
- 定理一: 如果 $\omega^*$ 是原问题的解， $\alpha^*, \ \beta^* $是对偶问题的解则有: $f(\omega^*) \ \ge \ \theta(\alpha^*, \ \beta^*) $
    - 对偶差距： $f(\omega^*) \ - \ \theta(\alpha^*, \ \beta^*) $
    - 强对偶定理： 如果 $g(\omega)=A\omega+b, \ h(\omega) = C\omega + d, \ f(\omega)$为凸函数，则有 $f(\omega^*)=\theta(\alpha^*, \ \beta^*) $, 则对偶差距为0
    - 根据`定理一`推出的不等式： 若 $f(\omega^*)=\theta(\alpha^*, \ \beta^*) $，则定理一中必然能够推出，对于所有的 i = 1 ~ K，要么 $\alpha_i=0$，要么 $g_i(\omega^*)=0$。这个条件称为KKT条件。

### 支持向量机 (转化为对偶问题)
1. 首先 $\delta_i \ge 0, \quad (i=1, \ \cdots, \ N)$ 转换成 $\delta_i \le 0, \quad (i=1, \ \cdots, \ N)$
2. 得到 $\frac{1}{2}{\Vert\omega\Vert}^2 - C\displaystyle\sum_{i=1}^N \delta_i $
3. 限制条件：$(1) \quad \delta_i \le 0, \quad (i=1, \ \cdots, \ N)$  
$\quad \quad \quad \ \ \ \  (2) \quad y_i[\omega^T\phi(X_i) + b] \ge 1 + \delta_i, \quad (i=1, \ \cdots, \ N)$
4. 再整理一下  
最小化(Minimize)：$\frac{1}{2}{\Vert\omega\Vert}^2 - C\displaystyle\sum_{i=1}^N \delta_i \ 或 \ \frac{1}{2}{\Vert\omega\Vert}^2 + C\displaystyle\sum_{i=1}^N \delta_i^2 \quad $ C为超参数，由人工赋值  
限制条件：$(1) \quad \delta_i \le 0, \quad (i=1, \ \cdots, \ N)$  
$\quad \quad \quad \ \ \ \  (2) \quad 1 + \delta_i - y_i \omega^T \phi(X_i) - y_ib \le 0, \quad (i=1, \ \cdots, \ N)$
5. 将对偶问题写成如下形式：  
最大化：$\theta(\alpha, \ \beta) = \displaystyle\max_{\omega, \ \delta_i, \ b} \{ \frac{1}{2} {\Vert\omega\Vert}^2 -C\displaystyle\sum_{i=1}^N \beta_i \delta_i + \displaystyle\sum_{i=1}0^N \alpha_i [1 + \delta_i - y_i \omega^T \phi(X_i) - y_i b] \} $  
限制条件：$(1) \quad \alpha_i \ge 0$  
$\quad \quad \quad \ \ \ \  (2) \quad \beta_i \ge 0$

### 如何化为对偶问题
1. 对$(\omega, \ b, \ \delta_i)$求导并令`导数为0`

### 将支持向量机的原问题华为对偶问题：
最大化：$\theta(\alpha, \ \beta) = \displaystyle\sum_{i=1}^N \alpha_i - \frac{1}{2}\displaystyle\sum_{i=1}^N \displaystyle\sum_{j=1}^N y_i y_j \alpha_i \alpha_j \phi(X_i)^T \phi(X_j) $  
限制条件：$(1) \quad0 \le \alpha_i \le 0, \quad (i \in \mathbb{N^*})$  
$\quad \quad \quad \ \ \ \  (2) \quad \displaystyle\sum_{i=1}^N \alpha_i y_i = 0, \quad (i \in \mathbb{N^*})$

### 算法流程


### 总结支持向量机训练和测试的流程
`训练过程`: 输入训练数据$ \{(X_i, \ y_i) i \in \mathbb{N^*}, \ y_i = +1 \ 或 \ -1 \} $  
最大化：$\theta(\alpha) = \displaystyle\sum_{i=1}^N \alpha_i - \frac{1}{2}\displaystyle\sum_{i=1}^N \displaystyle\sum_{j=1}^N y_i y_j \alpha_i \alpha_j \phi(X_i)^T \phi(X_j) $  
限制条件：$(1) \quad0 \le \delta_i \le C, \quad (i \in \mathbb{N^*})$  
$\quad \quad \quad \ \ \ \  (2) \quad \displaystyle\sum_{i=1}^N \alpha_i y_i = 0, \quad (i \in \mathbb{N^*})$  
测试过程：考察测试数据X，预测他的类别y  
如果 $ \displaystyle\sum_{i=1}^N \alpha_i y_i K(X_i, \ X) + b \ge 0, \ 则 \ y=+1 $  
如果 $ \displaystyle\sum_{i=1}^N \alpha_i y_i K(X_i, \ X) + b < 0, \ 则 \ y=-1 $

### 兵王问题描述
兵 （黑白各8个）：第一步向前可走一格或两格，以后每次只能向前走一格，不能后退。但在吃对方子时，则是向位于斜前方的那格去吃，并落在那个格。  
王 （黑白各1个）：是国际象棋中最为重要的棋子，王被将死即告负。走法是每次横直斜走均可，但每次只能走一格。吃子与走法相同。  
兵王问题：黑方只剩一个王，白方剩一个兵一个王。  
两种可能
- 白方将死黑方，获胜。
- 和棋。
这两种可能视三个棋子在棋盘的位置而确定。

兵的升变：兵走至对方底线，可以升变为除王以外的任意一子。  
逼和：一方的王未被将军，但移动到任意地方都会被对方将死，则此时是和棋。

### 兵王问题 程序参数设置
1. 对数据的预处理
- 总样本数为：`28056`，其中正样本`2796`，负样本`25260`。
- 随机取`5000`个样本训练，其余测试。
- 对训练样本归一化
    - 在训练样本上，求出每个维度的`均值`和`方差`，在训练和测试样本上同时`归一化`。
$$ newX = \frac{X-mean(X)}{std(X)} $$
2. 设置支持向量机的各种参数
    1. –s 0 "-s svm_type : set type of SVM (default 0)"
        - "   0 -- C-SVC      (multi-class classification)"
        - "   1 -- nu-SVC     (multi-class classification)"
        - "   2 -- one-class SVM"
        - "   3 -- epsilon-SVR    (regression)"
        - "   4 -- nu-SVR     (regression)"
    2. –t 2 "-t kernel_type : set type of kernel function (default 2)"
        - "   0 -- linear(线性内核): u'*v"
        - "   1 -- polynomial(多项式内核): (gamma*u'*v + coef0)^degree"
        - "   2 -- radial basis function(高斯径向基函数核): exp(-gamma*|u-v|^2)"
        - "   3 -- sigmoid(sigmoid核): tanh(gamma*u'*v + coef0)"
        - "   4 -- precomputed kernel(自定义核) (kernel values in training_instance_matrix)"
    3. –c CVALUE (C的值)
        - "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)"
    4. –g gammaValue (与核函数对应)
        - "-g gamma : set gamma in kernel function (default 1/num_features)"
    5. –v 5
        - "-v n : n-fold cross validation mode"

### 兵王问题 Matlab程序
```matlab
clear all;
% Read the data. 读文件
fid  =  fopen('krkopt.DATA');
c = fread(fid, 3);

vec = zeros(6,1);
xapp = []; % 接收数据
yapp = []; % 接收对应标签
while ~feof(fid)
    string = [];
    c = fread(fid,1);
    flag = flag+1;
    while c~=13
        string = [string, c];
        c=fread(fid,1);
    end;
    fread(fid,1);  
    if length(string)>10
        vec(1) = string(1) - 96;
        vec(2) = string(3) - 48;
        vec(3) = string(5) - 96;
        vec(4) = string(7) - 48;
        vec(5) = string(9) - 96;
        vec(6) = string(11) - 48;
        xapp = [xapp,vec];
        if string(13) == 100
            yapp = [yapp,1];
        else
            yapp = [yapp,-1];
        end;
    end;
end;
fclose(fid);

[N,M] = size(xapp);
p = randperm(M); %直接打乱了训练样本
numberOfSamplesForTraining = 5000;
% 获取训练集
xTraining = [];
yTraining = [];
for i=1:numberOfSamplesForTraining
    xTraining  = [xTraining,xapp(:,p(i))];
    yTraining = [yTraining,yapp(p(i))];
end;
xTraining = xTraining';
yTraining = yTraining';
% 获取测试集
xTesting = [];
yTesting = [];
for i=numberOfSamplesForTraining+1:M
    xTesting  = [xTesting,xapp(:,p(i))];
    yTesting = [yTesting,yapp(p(i))];
end;
xTesting = xTesting';
yTesting = yTesting';

%%%%%%%%%%%%%%%%%%%%%%%%
%Normalization
[numVec,numDim] = size(xTraining);
avgX = mean(xTraining);
stdX = std(xTraining);
% 训练集归一化操作
for i = 1:numVec
    xTraining(i,:) = (xTraining(i,:)-avgX)./stdX;
end;
[numVec,numDim] = size(xTesting);
% 测试集归一化操作
for i = 1:numVec
    xTesting(i,:) = (xTesting(i,:)-avgX)./stdX;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SVM Gaussian kernel 
%Search for the optimal C and gamma, K(x1,x2) = exp{-||x1-x2||^2/gamma} to
%make the recognition rate maximum. 

% Firstly, search C and gamma in a crude scale 
% (as recommended in 'A practical Guide to Support Vector Classification'))
CScale = [-5, -3, -1, 1, 3, 5,7,9,11,13,15];  % C的取值范围 2^-5 ~ 2^15 %
gammaScale = [-15,-13,-11,-9,-7,-5,-3,-1,1,3]; % gamma的取值范围 2^-15 ~ 2^3 %
C = 2.^CScale;
gamma = 2.^gammaScale;
maxRecognitionRate = 0;
for i = 1:length(C)
    for j = 1:length(gamma)
        cmd=['-t 2 -c ',num2str(C(i)),' -g ',num2str(gamma(j)),' -v 5']; % 5折交叉验证
        recognitionRate = svmtrain(yTraining,xTraining,cmd);
        if recognitionRate>maxRecognitionRate
            maxRecognitionRate = recognitionRate
            maxCIndex = i;
            maxGammaIndex = j;
        end;
    end;
end;

%Then search for optimal C and gamma in a refined scale. 
n = 10;
minCScale = 0.5*(CScale(max(1,maxCIndex-1))+CScale(maxCIndex));
maxCScale = 0.5*(CScale(min(length(CScale),maxCIndex+1))+CScale(maxCIndex));
newCScale = [minCScale:(maxCScale-minCScale)/n:maxCScale];

minGammaScale = 0.5*(gammaScale(max(1,maxGammaIndex-1))+gammaScale(maxGammaIndex));
maxGammaScale = 0.5*(gammaScale(min(length(gammaScale),maxGammaIndex+1))+gammaScale(maxGammaIndex));
newGammaScale = [minGammaScale:(maxGammaScale-minGammaScale)/n:maxGammaScale];
newC = 2.^newCScale;
newGamma = 2.^newGammaScale;
maxRecognitionRate = 0;
for i = 1:length(newC)
    for j = 1:length(newGamma)
        cmd=['-t 2 -c ',num2str(newC(i)),' -g ',num2str(newGamma(j)),' -v 5'];
        recognitionRate = svmtrain(yTraining,xTraining,cmd);
        if recognitionRate>maxRecognitionRate
            maxRecognitionRate = recognitionRate
            maxC = newC(i);
            maxGamma = newGamma(j);
        end;
    end;
end;

%Train the SVM model by the optimal C and gamma.
cmd=['-t 2 -c ',num2str(maxC),' -g ',num2str(maxGamma)];
model = svmtrain(yTraining,xTraining,cmd);
save model.mat model;

%Test the model on the remaining testing data and obtain the recognition rate.
load model.mat;
[yPred,accuracy,decisionValues] = svmpredict(yTesting,xTesting,model); 
save yPred.mat yPred;
save decisionValues.mat decisionValues;
save xTraining.mat xTraining;
save yTesting.mat yTesting;

%%draw ROC
[totalScores,index]  = sort(decisionValues);
labels = yTesting;
for i = 1:length(labels)
    labels(i) = yTesting(index(i));
end;


truePositive = zeros(1,length(totalScores)+1);
trueNegative = zeros(1,length(totalScores)+1);
falsePositive = zeros(1,length(totalScores)+1);
falseNegative = zeros(1,length(totalScores)+1);

for i = 1:length(totalScores)
    if labels(i) == 1
        truePositive(1) = truePositive(1)+1;
    else
        falsePositive(1) = falsePositive(1) +1;
    end;
end;

for i = 1:length(totalScores)
   if labels(i) == 1
       truePositive(i+1) = truePositive(i)-1;
       falsePositive(i+1) = falsePositive(i);
   else
       falsePositive(i+1) = falsePositive(i)-1;
       truePositive(i+1) = truePositive(i);
   end;
end;
truePositive = truePositive/truePositive(1);
falsePositive = falsePositive/falsePositive(1);
plot(falsePositive,truePositive);
```
结果如下所示: 
```matlab
K>> maxC % 查看最优C的值
maxC =
   1.1763e+03
K>> maxGamma % 查看最优Gamma的值
maxGamma =
    0.0625
K>> maxRecognitionRate % 查看训练模型的最优识别率
maxRecognitionRate =
   99.4800
```

### 支持向量机 (识别系统的性能度量)
- 混淆矩阵
    - TP: 将正样本识别为正样本的数量（或概率）
    - FN: 将正样本识别为负样本的数量（或概率）
    - FP: 将负样本识别为正样本的数量（或概率）
    - TN: 将负样本识别为负样本的数量（或概率）
- ROC曲线
    - TP+ -> FP+ 
    - TN- -> FN-
- AUC
- EER 等错误率

### 支持向量机 (多类情况)
1. 1类对K-1类  
假设总共有K类，我们需要构造K个支持向量机模型  
假设：  
对于每个优化问题，左边单一类别的标签为+1，右边K-1个类别的标签为-1。
$$ \{ \alpha_i^{(K)}\}_{i \in N*}, \ b^{(k)} \quad k = 1, \ \cdots, \ K $$
对于一个测试样本X，我们判断其类别为：  
$$ k_max = arg \max \displaystyle\sum_{i=1}^N \alpha_i^{(k)} y_i K(X_i, \ X) + b^{(k)}, \quad k \ = \ 1, \ \cdots, \ K $$
2. 1类对另一类  
聚类算法、结合决策树算法