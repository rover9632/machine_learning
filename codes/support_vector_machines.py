import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

import metrics

flags.DEFINE_bool("do_train", default=False, help="do training the model")
flags.DEFINE_bool("do_resume",
                  default=False,
                  help="resume latest checkpoint and continue train the model")
flags.DEFINE_bool("do_eval", default=False, help="do evaluation")
flags.DEFINE_bool("do_predict", default=False, help="do prediction")

FLAGS = flags.FLAGS


class SVC():
    """
        y_pred = sign(w^T · φ(x) + b)

    primal problem:
            min_{w,b,ζ} {1/2 * (w^T · w) + c * Σ_1^n ζ_i}
      s.t.  y_i * (w^T · φ(x_i) + b) >= 1 - ζ_i
            ζ_i >= 0, i = 1,...,n

    dual problem:
            min_α {1/2 * (α^T · Q · α) - e^T · α}
      s.t.  y^T · α = 0
            0 <= α_i <= c, i = 1,...,n

    where e is the vector of all ones, and Q is an n by n positive
    semidefinite matrix. Q_ij = y_i * y_j * K(x_i, x_j),
    where K(x_i, x_j) = φ(x_i)^T · φ(x_j) is the kernel.

    from dual perspect:
        y_pred = sign(Σ_i^nsv {y_i * α_i * K(x_i, x)} + b)

    where nsv is the number of support vectors.
    """

    def __init__(self, c=1.0, kernel="rbf", gamma=None, degree=3, coef0=0.0):
        self.c = c
        self.kernel = getattr(self, kernel)
        self.dual_coef = None
        self.support_vectors = None
        self.intercept = None
        self.kargs = {"gamma": gamma, "degree": degree, "coef0": coef0}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_2d = np.expand_dims(y, axis=-1)

        if self.kargs["gamma"] is None:
            self.kargs["gamma"] = 1.0 / n_features

        # standard quadratic programming form:
        # min {1/2 * x^T · P · x + q^T · x}
        #   s.t.  G · x <= h
        #         A · x = b
        P = matrix(np.dot(y_2d, y_2d.T) * self.kernel(X, X, **self.kargs))
        q = matrix(-np.ones((n_samples, 1)))
        G_0 = -np.eye(n_samples)
        G_1 = np.eye(n_samples)
        G = matrix(np.concatenate((G_0, G_1), axis=0))
        h_0 = np.zeros((n_samples, 1))
        h_1 = self.c * np.ones((n_samples, 1))
        h = matrix(np.concatenate((h_0, h_1), axis=0))
        A = matrix(y_2d.T)
        b = matrix([0.0])

        result = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(result["x"])

        indices = np.squeeze(alpha) > 1.0e-7
        self.dual_coef = y_2d[indices] * alpha[indices]
        self.support_vectors = X[indices]
        for i, x in enumerate(indices):
            if x and alpha[i] < self.c - 1.0e-7:
                kernel_mat = self.kernel(self.support_vectors, X[i:i + 1],
                                         **self.kargs)
                self.intercept = y[i] - np.sum(self.dual_coef * kernel_mat)
                break

        acc = self.accuracy_fn(X, y)
        print("training accuracy: %.4f" % acc)

    def accuracy_fn(self, X, y):
        preds = self.predict(X)
        return metrics.accuracy(y, preds)

    def evaluate(self, X, y):
        accuracy = self.accuracy_fn(X, y)
        return {"accuracy": accuracy}

    def predict(self, X):
        kernel_mat = self.kernel(self.support_vectors, X, **self.kargs)
        y_preds = np.sum(self.dual_coef * kernel_mat, axis=0) + self.intercept
        return np.sign(y_preds)

    def linear(self, X_i, X_j, **kwargs):
        return np.dot(X_i, X_j.T)

    def polynomial(self, X_i, X_j, gamma=0.1, degree=3, coef0=0.0, **kwargs):
        return (gamma * np.dot(X_i, X_j.T) + coef0)**degree

    def poly(self, X_i, X_j, gamma=0.1, degree=3, coef0=0.0, **kwargs):
        return self.polynomial(X_i, X_j, gamma, degree, coef0)

    def rbf(self, X_i, X_j, gamma=0.1, **kwargs):
        X_i = np.expand_dims(X_i, axis=1)
        X_j = np.expand_dims(X_j, axis=0)
        return np.exp(-gamma * np.sum(np.square(X_i - X_j), axis=-1))

    def sigmoid(self, X_i, X_j, gamma=0.001, coef0=0.0, **kwargs):
        return np.tanh(gamma * np.dot(X_i, X_j.T) + coef0)

    def save(self, model_path):
        params = {
            "inits": {
                "c": self.c,
                "kernel": self.kernel.__name__
            },
            "attrs": {
                "dual_coef": self.dual_coef,
                "support_vectors": self.support_vectors,
                "intercept": self.intercept,
                "kargs": self.kargs
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

    y = np.where(y == 1.0, 1., -1.0)
    X = df.drop(columns=[4]).values

    return X, y


def main(_):
    model_path = "./models/svm.pkl"

    if FLAGS.do_train:
        model = SVC(kernel="rbf")
        data_path = "../datasets/banknote_auth/train.csv"
        X_train, y_train = prepare_data(data_path, is_training=True)

        model.fit(X_train, y_train)
        model.save(model_path)

    if FLAGS.do_eval:
        data_path = "../datasets/banknote_auth/dev.csv"
        X_dev, y_dev = prepare_data(data_path, is_training=False)

        model = SVC.restore(model_path)
        result = model.evaluate(X_dev, y_dev)
        print(result)


if __name__ == "__main__":
    app.run(main)
