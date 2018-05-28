###Introduction
**Machine learning** is a field of computer science that gives computers the ability to learn without being explicitly programmed. Human can learn from past experiences, so do the computers, for computers past experiences are just recorded as data. 

---

###Data
- **structured data** :  data which is organized. (e.g. students infomation including name, age, gender, etc.)
- **unstructured data** : opposite of structured data. (e.g. texts and images)
- **data type** : continuous number, categories, etc.

---

###Data Preparation
- **data cleaning** : deal with missing values, noisy data and outliers.
- **feature encoding** : non-numeric data must translate to numeric data.
  - **label encoding** : e.g. ['male', 'female'] represented as [0, 1].
  - **one hot encoding** : e.g. ['male', 'female'] represented as [[1,0], [0,1]].
- **standardization** : transform to have zero mean and unit variance.
- **scaling** : scaling to [-1, 1] or [0, 1].
- **normalization** : scaling individual samples to have unit norm.
- **feature selection** : select useful features.


---

###Model
A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.

---

###Task
- **Supervised learning** : learning with label.
- **Unsupervised learning** : learning without label.
- **Classification** : the outputs are discrete. 
- **Regression** : the outputs are continuous.

---

###Training Model
- **loss/cost/objective function** : 
