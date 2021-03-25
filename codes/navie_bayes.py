import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd

import metrics

flags.DEFINE_bool("do_train", default=False, help="do training the model")
flags.DEFINE_bool("do_resume",
                  default=False,
                  help="resume latest checkpoint and continue train the model")
flags.DEFINE_bool("do_eval", default=False, help="do evaluation")
flags.DEFINE_bool("do_predict", default=False, help="do prediction")

FLAGS = flags.FLAGS


class GaussianNB():
    """
    y_pred = argmax_c P(c) ∏_1^n P(x_i|c)
    where
    P(x_i|c) = exp(-0.5 * (x_i - mu_c)^2 / sigma_c^2) / sqrt(2 * pi * sigma_c^2)
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.mu = None
        self.variance = None
        self.count = None
        self.prior = None

    def fit(self, X, y, batch_size=32):
        n_samples, n_features = X.shape

        self.mu = np.zeros(shape=(self.n_classes, n_features))
        self.variance = np.zeros(shape=(self.n_classes, n_features))
        self.count = np.zeros(shape=self.n_classes)
        self.prior = np.zeros(shape=self.n_classes)

        for i in range(0, n_samples, batch_size):
            X_batch, y_batch = X[i:i + batch_size], y[i:i + batch_size]
            self.train_step(X_batch, y_batch)

        self.prior = self.count / np.sum(self.count)

        acc = self.accuracy_fn(X, y)
        print("training accuracy: %.4f" % acc)

    def train_step(self, X, y):
        """Online update of Gaussian mean and variance.

        for variance updating formula details, see:
        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        """
        for i in np.unique(y):
            X_i = X[y == i, :]
            b_mu = np.mean(X_i, axis=0)
            b_var = np.var(X_i, axis=0)
            b_count = X_i.shape[0]
            n_total = self.count[i] + b_count

            mu = (self.count[i] * self.mu[i] + b_count * b_mu) / n_total
            part1 = self.count[i] * self.variance[i] + b_count * b_var
            part2 = (self.count[i] * b_count / n_total) * (self.mu[i] - b_mu)**2
            variance = (part1 + part2) / n_total

            self.mu[i, :] = mu
            self.variance[i, :] = variance
            self.count[i] = n_total

    def accuracy_fn(self, X, y):
        preds = self.predict(X)
        return metrics.accuracy(y, preds)

    def evaluate(self, X, y):
        accuracy = self.accuracy_fn(X, y)
        return {"accuracy": accuracy}

    def predict(self, X):
        if self.mu is None:
            raise Exception("Not trained yet !")

        X = np.expand_dims(X, axis=1)
        mu = np.expand_dims(self.mu, axis=0)
        variance = np.expand_dims(self.variance, axis=0)

        log_likelihood = -0.5 * np.square(X - mu) / variance
        log_likelihood -= 0.5 * np.log(2 * np.pi * variance)
        joint_log_likelihood = np.log(self.prior) + np.sum(log_likelihood, -1)

        return np.argmax(joint_log_likelihood, axis=-1)

    def save(self, model_path):
        params = {
            "inits": {
                "n_classes": self.n_classes
            },
            "attrs": {
                "mu": self.mu,
                "variance": self.variance,
                "count": self.count,
                "prior": self.prior
            }
        }
        with open(model_path, "wb") as f:
            pickle.dump(params, f)

    @classmethod
    def restore(cls, model_path):
        with open(model_path, "rb") as f:
            params = pickle.load(f)

        model = cls(**params["inits"])
        for k, v in params["attrs"].items():
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
    model_path = "./models/gaussian_naive_bayes.pkl"

    if FLAGS.do_train:
        data_path = "../datasets/iris/train.csv"
        X_train, y_train = prepare_data(data_path, label_map, is_training=True)

        model = GaussianNB(n_classes=len(label_map))
        model.fit(X_train, y_train, batch_size=128)
        model.save(model_path)

    if FLAGS.do_eval:
        data_path = "../datasets/iris/dev.csv"
        X_dev, y_dev = prepare_data(data_path, label_map, is_training=False)

        model = GaussianNB.restore(model_path)
        result = model.evaluate(X_dev, y_dev)
        print(result)


if __name__ == "__main__":
    app.run(main)
