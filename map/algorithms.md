### Linear Models
The representation of linear regression is an equation that describes a line that best fits the relationship between the input variables (x) and the target variables (y), by finding specific weightings for the input variables.  That is the target value is expected to be a linear combination of the input variables.

#### Ordinary Least Squares
Fits a linear model with coefficients to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.

#### Ridge Regression
Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty (here, i.e. L2-norm regularization) on the size of coefficients.

#### Lasso
The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent.

#### Logistic Regression
Logistic regression, despite its name, is a linear model for binary classification rather than regression. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic/sigmoid function.

The predict function is:

$$
\hat y(x) = P(y=1|x;\theta ) = sigmoid(\theta_0 + \theta_1 x_1 + ... + \theta_p x_p)
$$

where $sigmoid(z)$ is [sigmoid function](../concepts/maths.md).

The loss function is:

$$
L = -[ y \log {\hat y} + (1 - y) \log(1 - \hat y) ]
$$

特点：  
1. 属于判别式模型 (Discriminative Model) 。
2. 适用于分类任务，原生是二分类的，采用 One-vs-Rest 等方法可支持多分类任务。
3. 仅适用于线性问题，即样本的特征空间是线性可分的。
4. 输出结果在0到1之间，有概率意义。
5. 实现简单，计算量小，速度快，存储资源低。
6. 可以处理特征间有相关性的问题。
7. 当特征空间很大时，性能不是很好。
8. 容易欠拟合，一般准确度不是很高。

#### Linear Discriminant Analysis (LDA)
The representation of LDA is pretty straight forward. For a given dataset, mapping the data points on a line,  making the mapping points of those which belong to same class as close as possible and the mapping points of those which belong to different class as far as possible. Prediction is made by mapping a new sample on the same line and finding which class it closest to.

### Decision Trees
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.  

Given training vectors $x_i \in R^n ,\; i=1,...,l$ and a label vector $y \in R^l$ , a decision tree recursively partitions the space such that the samples with the same labels are grouped together.  

Let the data at node $m$ be represented by $Q_m$ with $N_m$ samples. For each  candidate split $\theta = (j, t_m)$ (called condition $A_m$) consisting of a feature $j$ and threshold $t_m$ , partition the data into $Q_m^{left}(\theta)$ and $Q_m^{right}(\theta)$ subsets :

$$
Q_m^{left}(\theta) = \{(x, y) | x_j <= t_m\} \;\;and\;\; ​Q_m^{right}(\theta) = Q_m \setminus Q_m^{left}(\theta)
$$

Define an impurity function or loss function $H()$ , then the impurity of split (conditioned $A_m$) is :

$$
H(Q_m \mid A_m) = \frac{N_m^{left}}{N_m} H(Q_m^{left}(\theta))
+ \frac{N_m^{right}}{N_m} H(Q_m^{right}(\theta))
$$

The information gain is :

$$
G(Q_m, A_m) = H(Q_m) - H(Q_m \mid A_m)
$$

Then select the split that maximize the information gain :

$$
\theta^* = \operatorname{argmax}_\theta  G(Q_m, A_m)
$$

Recurse for subsets $Q_m^{left}(\theta^*)$ and $Q_m^{right}(\theta^*)$ until the maximum allowable depth is reached or non-splittable or information gain is 0.

If a target is a classification outcome taking on values $0,1,…,K-1$, for node $m$, let

$$
p_{mk} = \frac{1}{N_m} \sum_{y \in Q_m} I(y = k)
$$

be the proportion of class $k$ observations in node $m$. Common measures of impurity are the following :

Gini :

$$
H(Q_m) = \sum_k p_{mk} (1 - p_{mk})
$$

Entropy :

$$
H(Q_m) = - \sum_k p_{mk} \log(p_{mk})
$$

特点：
1. 容易理解和解释，可以可视化。
2. 只需要很少的数据准备。
3. 预测快，复杂度是决策树的深度。
4. 能够处理数字类型(numerical)和分类类型(categorical)的数据。
5. 能够处理多输出问题。
6. 使用白箱模型，可以通过布尔逻辑轻松解释。而在黑箱模型（如神经网络）中，可能难以解释。
7. 可以使用统计测试来验证模型。这使得考虑模型的可靠性成为可能。
8. 即使生成数据的真实模型在某种程度上违背了它的假设，也可以表现良好。
9. 容易过拟合，需要通过剪枝、设置最大深度等机制来避免此问题。
10. 可能不稳定，因为数据中的细微变化可能会导致生成完全不同的树。通过集成方法可以缓解此问题。
11. 决策树的预测既不是平滑的也不是连续的，而是分段恒定的近似值。因此，他们不擅长外推。
12. 最优学习被认为是NP完备问题(NP-complete)。因此，一般基于启发式算法加上集成方法来缓解。
13. 有些概念很难学习，因为决策树无法轻松表达它们，例如XOR，奇偶校验或多路复用器问题。
14. 如果某些类别占主导，则容易偏向它们，因此要做数据平衡。


### Naive Bayes
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.

The predict function is:

$$
\hat{y} = \arg\max_c P(c) \prod_{i=1}^{n} P(x_i \mid c)
$$

where $c$ means class or category.

For gaussian naive bayes, the likelihood of the features is assumed to obey gaussian distribution. That is:

$$
P(x_i \mid c) = \frac{1}{\sqrt{2\pi\sigma^2_c}} \exp\left(-\frac{(x_i - \mu_c)^2}{2\sigma^2_c}\right)
$$

For multinomial naive bayes, the likelihood of the features is assumed to obey multinomial distribution. That is (smoothed version):

$$
P(x_i \mid c) = \frac{ N_{ci} + \alpha}{N_c + \alpha n}
$$

where $n$ is the number of features, $N_{ci} = \sum_{x \in T} x_i$ is the number of times feature $i$ appears in a sample of class $c$ in the training set $T$, and $N_{c} = \sum_{i=1}^{n} N_{ci}$ is the total count of all features for class $c$.  
ps: here *feature* actually means unique feature column value for one specific feature column. for example, feature column *gender* have value of *man* and *woman*, here *feature* are *man* or *woman*, not *gender*.

The smoothing priors $\alpha \ge 0$ accounts for features not present in the learning samples and prevents zero probabilities in further computations. Setting $\alpha = 1$ is called Laplace smoothing, while $\alpha < 1$ is called Lidstone smoothing.

特点：  
1. 属于生成式模型 (Generative Model) 。
2. 适用于多分类任务。
3. 依赖于特征间条件独立性假设。
4. 实现简单，计算高效。
5. 特征间相关性较大时，分类效果不怎么好。


### K-Nearest Neighbors
Predictions are made for a new data point by searching through the entire training set for the K most similar instances (the neighbors) and summarizing the output variable for those K instances. For regression problems, this might be the mean output variable, for classification problems this might be the mode (or most common) class value.

### Support Vector Machines
A support vector machine (SVM) constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification, regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier. Only these nearest training data points are relevant in defining the hyperplane and in the construction of the classifier. These points are called the support vectors.

Given training vectors $x_i \in \mathbb{R}^p \;,\; i=1,...,n$ , in two classes, and a vector $y \in \{1, -1\}^n$ , our goal is to find $w \in \mathbb{R}^p$ and $b \in \mathbb{R}$ , such that the prediction given by $\text{sign} (w^T x + b)$ is correct for most samples.

The hyper-plane would be $w^T x + b = 0$, let the minimal gap be $\gamma$, for a perfect separation, the problem is :

$$
\begin{aligned}
\max_{w,b} \quad & \gamma \\
\text{s.t.} \quad & \frac{y_i \; (w^T x_i + b)}{||w||} \geq \gamma \;,\; i=1,2,...,n
\end{aligned}
$$

Since $w$ and $b$ changed proportionally, the hyper-plane remain unchanged, let $||w|| \gamma = 1$, then the problem is :

$$
\begin{aligned}
\min_{w, b, \zeta} \quad & \frac{1}{2} w^T w \\
\text{s.t.} \quad & y_i (w^T x_i + b) \geq 1 \;,\; i=1,2, ..., n
\end{aligned}
$$

To soft the margin with a distance $\zeta_i$ for each sample and add a penalty item $C \sum_{i=1}^{n} \zeta_i$ to penal the samples those are misclassified or within the margin boundary, where the term $C$ controls the strength of this penalty, acts as an inverse regularization parameter. Then the primal problem is :

$$
\begin{aligned}
\min_{w, b, \zeta} \quad & \frac{1}{2} w^T w + C \sum_{i=1}^{n} \zeta_i \\
\text{s.t.} \quad & y_i (w^T x_i + b) \geq 1 - \zeta_i \;,\\
& \zeta_i \geq 0 \;,\; i=1,2, ..., n
\end{aligned}
$$

Construct the Lagrange function :

$$
\begin{aligned}
L(w,b,\zeta,\alpha,\lambda) = & \; \frac{1}{2} w^T w + C \sum_{i=1}^{n} \zeta_i + \sum_{i=1}^{n} \alpha_i [1 - \zeta_i - y_i (w^T x_i + b)] - \sum_{i=1}^{n} \lambda_i \zeta_i \;,\\
\text{s.t.} \quad & \; \alpha_i \geq 0 \;,\; \lambda_i \geq 0 \;,\; i=1,...,n
\end{aligned}
$$

Then the problem is :

$$
\min_{w, b, \zeta} \; \max_{\alpha, \lambda} \; L(w,b,\zeta,\alpha,\lambda)
$$

Exchange the $\min$ and $\max$, then got the dual problem :

$$
\max_{\alpha, \lambda} \; \min_{w, b, \zeta} \; L(w,b,\zeta,\alpha,\lambda)
$$

To solve the problem $\min_{w, b, \zeta} \; L(w,b,\zeta,\alpha,\lambda)$ , is to let the partial derivative of $L$ with respect to $w$ , $b$ and $\zeta$ equals 0 :

$$
\begin{aligned}
\frac {\partial L} {\partial w} = & \; w - \sum_{i=1}^{n} \alpha_i x_i y_i = 0 \\
\frac {\partial L} {\partial b} = & - \sum_{i=1}^{n} \alpha_i y_i = 0 \\
\frac {\partial L} {\partial \zeta_i} = & \; C - \alpha_i - \lambda_i = 0 \;,\; i=1,2,...,n
\end{aligned}
$$

got :

$$
\begin{aligned}
w = & \sum_{i=1}^{n} \alpha_i x_i y_i \\
\sum_{i=1}^{n} \alpha_i y_i = & \; 0 \\
\lambda_i = & \; C - \alpha_i \;,\; i=1,2,...,n \\
L(w,b,\zeta,\alpha,\lambda) = & - \frac {1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (x_i^T x_j) + \sum_{i=1}^{n} \alpha_i
\end{aligned}
$$

Next is to solve the problem $\max_{\alpha, \lambda} \; L(w,b,\zeta,\alpha,\lambda)$, considering the constraints $\alpha_i \geq 0 \;,\; \lambda_i \geq 0 \;,\; i=1,...,n$, the dual problem become :

$$
\begin{aligned}
\min_{\alpha} \quad & \frac{1}{2} \alpha^T Q \alpha - e^T \alpha \\
\text {s.t.} \quad & y^T \alpha = 0 \\
& 0 \leq \alpha_i \leq C \;,\; i=1,2, ..., n
\end{aligned}
$$

where $e$ is the vector of all ones, and $Q$ is an n by n positive semidefinite matrix , $Q_{ij} = y_i y_j (x_i^T x_j)$ .

If we replace all $x$ with $\phi (x)$ , then we get a kernelized SVM , where $Q_{ij} = y_i y_j K(x_i, x_j)$ , where $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ is the kernel.

This is a convex quadratic programming problem, it can be solved by sequential minimal optimization (SMO) .

For a given sample $x$, the prediction is :

$$
\hat y = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b
$$

where $b = y_j - \sum_{i \in SV} \alpha_i y_i K(x_i, x_j)$ , where $j$ is the index of one of the samples that satisfied $0 < \alpha_j < C$ , and $SV$ is the collection of indices that satisfied $\alpha_i > 0$, which the corresponding sample called support vector. 

Kernels :  
linear : $K(x_i, x_j) = x_i^T x_j$  
polynomial : $K(x_i, x_j) = (\gamma (x_i^T x_j) + r)^d$  
rbf : $K(x_i, x_j) = \text{exp} (-\gamma (x_i - x_j)^T (x_i - x_j))$  
sigmoid : $K(x_i, x_j) = \tanh(\gamma (x_i^T x_j) + r)$

特点：
1. 原生是二分类的，采用 One-vs-One 等方法可支持多分类任务。
2. 属于凸二次规划问题(convex quadratic programming problem)，局部最优解就是全局最优解。
3. 可以有效处理高维空间数据。
4. 特征维度大于样本数量时依然有效。
5. 只使用少量的训练数据子集(即支持向量)来决策，所以也节约内存。
6. 使用核技巧，也可以处理各种非线性可分的问题。
7. 特征维度远大于样本数量时容易过拟合，要注意核函数选择和正则化。
8. 不直接提供概率估计。
9. 训练数据量很大时非常消耗内存和训练时间。


### Ensemble methods
The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.

#### Averaging methods
The driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.

##### Bagging methods
Bagging methods form a class of algorithms which build several instances of a black-box estimator on random subsets of the original training set and then aggregate their individual predictions to form a final prediction.

##### Forests of randomized trees
They are two averaging algorithms based on randomized decision trees: the Random Forest algorithm and the Extra-Trees method. Both algorithms are perturb-and-combine techniques specifically designed for trees. This means a diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.

#### Boosting methods
Base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.

##### AdaBoost
The core principle of AdaBoost is to fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction.

##### Gradient Tree Boosting
Gradient Tree Boosting or Gradient Boosted Regression Trees (GBRT) is a generalization of boosting to arbitrary differentiable loss functions. GBRT is an accurate and effective off-the-shelf procedure that can be used for both regression and classification problems.

### Artificial Neural Network
see [deeplearning](./deeplearning.md)

### Clustering

#### K-means
The K-Means algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares. This algorithm requires the number of clusters to be specified.
