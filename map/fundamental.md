### Introduction
**Machine learning** is a field of computer science that gives computers the ability to learn without being explicitly programmed. Human can learn from past experiences, so do the computers, for computers past experiences are just recorded as data.

### Data
- **structured data** :  data which is organized. (e.g. students infomation including name, age, gender, etc.)
- **unstructured data** : opposite of structured data. (e.g. texts and images)
- **data type** : continuous number, categories, etc.

### Data Preparation
- **data cleaning** : deal with missing values, noisy data and outliers.
- **feature encoding** : non-numeric data must translate to numeric data.
  - **label encoding** : e.g. ['male', 'female'] represented as [0, 1].
  - **one hot encoding** : e.g. ['male', 'female'] represented as [[1,0], [0,1]].
- **standardization** : transform to have zero mean and unit variance.
- **scaling** : scaling to [-1, 1] or [0, 1].
- **normalization** : scaling individual samples to have unit norm.
- **feature selection** : select useful features.
- **Principal Component Analysis (PCA)** : used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance.

### Model
A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.

### Task
- **Supervised learning** : learning with labels.
- **Unsupervised learning** : learning without labels.
- **Classification** : the outputs are discrete.
- **Regression** : the outputs are continuous.

### Model Training
- **objective function** : ![](https://latex.codecogs.com/gif.latex?J=L+R) , where ![](https://latex.codecogs.com/gif.latex?L) is the loss function that measuring how far away from the correct results, and ![](https://latex.codecogs.com/gif.latex?R) is the regularization term that to reduce model complexity. Regularization term is not necessary.
  - **loss function** : cross-entropy loss (log loss), logistic loss, quadratic loss, 0-1 loss, hinge loss, mean squared error (MSE), mean absolute error (MAE), etc.
  - **regularization** : L1 norm, L2 norm, dropout, etc.
- **optimization** : minimise the objective function. Typically this is done by *Gradient Descent* or *Stochastic Gradient Descent*.

### Model Evaluation
- **Metrics for classification** : accuracy, precision, recall, F score, ROC, AUC, log loss, etc.
- **Metrics for regression** : MAE (mean absolute error), MSE (mean squared error), RMSE (root means squared error), MAPE (mean absolute percentage error), R^2 (coefficient of determination), etc.
- **Validation strategy** : Hold-out validation, K-fold cross validation, Leave-one-out cross Validation.

### Model Optimization
- **Bias–variance trade-off** : high bias means underfitting, high variance means overfitting.
  - **when underfitting** :
    - Adjust hyper-parameters to make the model more complex.
    - Get more features.
    - Change to more complex algorithm.
    - Use an ensemble of low complexity algorithms (Boosting).    
  - **when overfitting** :
    - Obtain more data.
    - Adjust hyper-parameters to make the model more simple.
    - Decrease features.
    - Use regularization.
    - Change to more simple algorithm.
    - Use an ensemble of high complex algorithms (Bagging).
- **Grid Search** : an exhaustive searching through a manually specified subset of the hyper-parameter space.
- **Random search** : select hyper-parameters randomly.

### Details
[metrics](../concepts/metrics.md)
