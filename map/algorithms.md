### Linear Models
The representation of linear regression is an equation that describes a line that best fits the relationship between the input variables (x) and the target variables (y), by finding specific weightings for the input variables.  That is the target value is expected to be a linear combination of the input variables.

#### Ordinary Least Squares
Fits a linear model with coefficients to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.

#### Ridge Regression
Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty (here, i.e. L2-norm regularization) on the size of coefficients.

#### Lasso
The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent.

#### Logistic Regression
Logistic regression, despite its name, is a linear model for binary classification rather than regression. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

#### Linear Discriminant Analysis (LDA)
The representation of LDA is pretty straight forward. For a given dataset, mapping the data points on a line,  making the mapping points of those which belong to same class as close as possible and the mapping points of those which belong to different class as far as possible. Prediction is made by mapping a new sample on the same line and finding which class it closest to.

### Decision Trees
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
Given training vectors and a label vector , a decision tree recursively partitions the space such that the samples with the same labels are grouped together.

### Naive Bayes
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features.

#### Gaussian Naive Bayes
The likelihood of the features is assumed to obey Gaussian distribution.

#### Multinomial Naive Bayes
The likelihood of the features is assumed to obey Multinomial distribution.

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
