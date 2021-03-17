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
1. 属于判别式模型 (Discrimitive Model) 。
2. 适用于分类任务，原生是二分类的，采用 One-vs-Rest 等方法可支持多分类任务。
3. 仅适用于线性问题，即样本的特征空间是线性可分的。
4. 输出结果在0到1之间，有概率意义。
5. 实现简单，计算量小，速度快，存储资源低。
6. 可以处理特征间有相关性的问题。
7. 当特征空间很大时，性能不是很好。
8. 容易欠拟合，一般准确度不很高。

#### Linear Discriminant Analysis (LDA)
The representation of LDA is pretty straight forward. For a given dataset, mapping the data points on a line,  making the mapping points of those which belong to same class as close as possible and the mapping points of those which belong to different class as far as possible. Prediction is made by mapping a new sample on the same line and finding which class it closest to.

### Decision Trees
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
Given training vectors and a label vector , a decision tree recursively partitions the space such that the samples with the same labels are grouped together.

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
A support vector machine constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification, regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier. Only these nearest training data points are relevant in defining the hyperplane and in the construction of the classifier. These points are called the support vectors.

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
