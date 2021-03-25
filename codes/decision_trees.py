import json
from collections import Counter
from absl import app
from absl import flags
from absl import logging
import numpy as np
from numpy.core.numeric import indices
import pandas as pd

import metrics

flags.DEFINE_bool("do_train", default=False, help="do training the model")
flags.DEFINE_bool("do_resume",
                  default=False,
                  help="resume latest checkpoint and continue train the model")
flags.DEFINE_bool("do_eval", default=False, help="do evaluation")
flags.DEFINE_bool("do_predict", default=False, help="do prediction")

FLAGS = flags.FLAGS


class Node():

    def __init__(self):
        self.child_left = None
        self.child_right = None
        self.feature = None
        self.threshold = None
        self.impurity = None
        self.n_samples = None
        self.type_ = None
        self.class_ = None

    def update_attrs(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("child_") and v is not None:
                result[k] = v.to_dict()
                continue
            if isinstance(v, (np.ndarray, np.number)):
                v = v.tolist()
            result[k] = v

        return result

    @classmethod
    def from_dict(cls, **kwargs):
        node = cls()
        for k, v in kwargs.items():
            if k.startswith("child_") and v is not None:
                v = cls.from_dict(**v)
            setattr(node, k, v)
        return node


class Criterion():

    def __init__(self, name=None) -> None:
        self.name = name

    def compute_impurity(self, y):
        raise NotImplementedError()

    def proxy_impurity_gain(self, y_left, y_right):
        impurity_left = self.compute_impurity(y_left)
        impurity_right = self.compute_impurity(y_right)
        return -(len(y_left) * impurity_left + len(y_right) * impurity_right)

    def impurity_gain(self, n_samples, y_node, y_left, y_right):
        impurity_node = self.compute_impurity(y_node)
        impurity_left = self.compute_impurity(y_left)
        impurity_right = self.compute_impurity(y_right)

        impurity_split = len(y_left) / len(y_node) * impurity_left
        impurity_split += len(y_right) / len(y_node) * impurity_right

        return len(y_node) / n_samples * (impurity_node - impurity_split)


class Entropy(Criterion):

    def __init__(self) -> None:
        super().__init__(name="entropy")

    def compute_impurity(self, y):
        impurity = 0.0
        n = len(y)
        for k in np.unique(y):
            p_k = np.sum(y == k) / n
            impurity += -(p_k * np.log(p_k))
        return impurity


class Gini(Criterion):

    def __init__(self) -> None:
        super().__init__(name="gini")

    def compute_impurity(self, y):
        impurity = 0.0
        n = len(y)
        for k in np.unique(y):
            p_k = np.sum(y == k) / n
            impurity += p_k * (1 - p_k)
        return impurity


class Splitter():

    def __init__(self, criterion, min_samples_leaf) -> None:
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf

    def split(self, X, y):
        n_samples, n_features = X.shape
        best_feature = None
        threshold = None
        best_impurity_gain = -np.inf

        for i in range(n_features):
            feature = X[:, i]
            indices = np.argsort(feature)
            sorted_feature = feature[indices]
            sorted_y = y[indices]
            for j in range(self.min_samples_leaf,
                           n_samples - self.min_samples_leaf):
                if sorted_feature[j] <= sorted_feature[j - 1] + 1e-7:
                    continue
                impurity_gain = self.criterion.proxy_impurity_gain(
                    sorted_y[:j], sorted_y[j:])
                if impurity_gain > best_impurity_gain:
                    best_feature = i
                    best_impurity_gain = impurity_gain
                    threshold = (sorted_feature[j - 1] + sorted_feature[j]) / 2

        return best_feature, threshold


class DecisionTree():

    def __init__(self,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1):
        self.criterion = Gini() if criterion == "gini" else Entropy()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.splitter = Splitter(self.criterion, min_samples_leaf)
        self.root = Node()

    def fit(self, X, y):
        self.generate_tree(self.root, X, y, 1)
        acc = self.accuracy_fn(X, y)
        print("accuracy: %.4f" % acc)

    def generate_tree(self, node, X, y, depth):
        n_samples = X.shape[0]
        impurity = self.criterion.compute_impurity(y)
        counter = Counter(y)

        if (self.max_depth and depth > self.max_depth or
                n_samples < self.min_samples_split or len(counter) == 1):
            class_ = sorted(counter.items(), key=lambda x: -x[1])[0][0]
            node.update_attrs(type_="leaf",
                              class_=class_,
                              impurity=impurity,
                              n_samples=n_samples)
            return

        feature, threshold = self.splitter.split(X, y)
        if feature is None:
            class_ = sorted(counter.items(), key=lambda x: -x[1])[0][0]
            node.update_attrs(type_="leaf",
                              class_=class_,
                              impurity=impurity,
                              n_samples=n_samples)
            return

        node.update_attrs(child_left=Node(),
                          child_right=Node(),
                          type_="node",
                          impurity=impurity,
                          n_samples=n_samples,
                          feature=feature,
                          threshold=threshold)

        ids = X[:, feature] <= threshold
        self.generate_tree(node.child_left, X[ids], y[ids], depth + 1)
        self.generate_tree(node.child_right, X[~ids], y[~ids], depth + 1)

    def accuracy_fn(self, X, y):
        preds = self.predict(X)
        return metrics.accuracy(y, preds)

    def evaluate(self, X, y):
        accuracy = self.accuracy_fn(X, y)
        return {"accuracy": accuracy}

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            cur_node = self.root
            while cur_node.type_ != "leaf":
                if X[i][cur_node.feature] <= cur_node.threshold:
                    cur_node = cur_node.child_left
                else:
                    cur_node = cur_node.child_right
            preds.append(cur_node.class_)

        return np.array(preds)

    def save(self, model_path):
        params = {
            "inits": {
                "criterion": self.criterion.name,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
            },
            "attrs": {
                "root": self.root.to_dict()
            }
        }
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    @classmethod
    def restore(cls, model_path):
        with open(model_path, encoding="utf-8") as f:
            params = json.load(f)

        model = cls(**params["inits"])
        for k, v in params["attrs"].items():
            if k == "root":
                v = Node.from_dict(**v)
            setattr(model, k, v)

        return model


def prepare_data(data_path, label_map=None, is_training=False):
    df = pd.read_csv(data_path, header=None)

    if is_training:
        df = df.sample(frac=1).reset_index(drop=True)

    if label_map:
        y = np.array(list(map(label_map.get, df[4])))
    else:
        y = df[4].values
    X = df.drop(columns=[4]).values

    return X, y


def main(_):
    label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    model_path = "./models/decision_tree.json"

    if FLAGS.do_train:
        data_path = "../datasets/iris/train.csv"
        X_train, y_train = prepare_data(data_path, label_map, is_training=True)

        model = DecisionTree(criterion="gini", max_depth=4)
        model.fit(X_train, y_train)
        model.save(model_path)

    if FLAGS.do_eval:
        data_path = "../datasets/iris/dev.csv"
        X_dev, y_dev = prepare_data(data_path, label_map, is_training=False)

        model = DecisionTree.restore(model_path)
        result = model.evaluate(X_dev, y_dev)
        print(result)


if __name__ == "__main__":
    app.run(main)
