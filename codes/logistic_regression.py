import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd

import initializers
import metrics
import maths

flags.DEFINE_bool("do_train", default=False, help="do training the model")
flags.DEFINE_bool("do_resume",
                  default=False,
                  help="resume latest checkpoint and continue train the model")
flags.DEFINE_bool("do_eval", default=False, help="do evaluation")
flags.DEFINE_bool("do_predict", default=False, help="do prediction")

FLAGS = flags.FLAGS


class LogisticRegression():
    """ p(x) = g(weights · x + bias)
        where g(z) = 1 / (1 + exp(-z))
    """

    def __init__(self, alpha=0.001):
        self.weights = None
        self.bias = 0.0
        self.alpha = alpha

    def fit(self, X, y, epochs=1, batch_size=32):
        n_samples, n_features = X.shape
        self.weights = initializers.glorot_uniform((n_features,))

        for epoch in range(1, epochs + 1):
            for i in range(0, n_samples, batch_size):
                X_batch, y_batch = X[i:i + batch_size], y[i:i + batch_size]
                self.train_step(X_batch, y_batch)

            loss = self.loss_fn(X, y)
            acc = self.accuracy_fn(X, y)
            print("Epoch: %d, loss: %.4f, acc: %.4f" % (epoch, loss, acc))

    def train_step(self, X, y):
        self.optimize(X, y)

    def optimize(self, X, y):
        probs = self.predict(X)

        # the gradients with respect to z = weights · x + bias
        linear_gradients = -(y * (1.0 - probs) - (1.0 - y) * probs)

        weight_gradients = np.expand_dims(linear_gradients, axis=1) * X
        weight_gradients = np.mean(weight_gradients, axis=0)
        bias_gradient = np.mean(linear_gradients)

        self.weights -= self.alpha * weight_gradients
        self.bias -= self.alpha * bias_gradient

    def loss_fn(self, X, y):
        probs = self.predict(X)
        return np.mean(-(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))

    def accuracy_fn(self, X, y):
        probs = self.predict(X)
        preds = (probs > 0.5).astype(np.int32)
        return metrics.accuracy(y, preds)

    def evaluate(self, X, y):
        loss = self.loss_fn(X, y)
        accuracy = self.accuracy_fn(X, y)
        return {"loss": loss, "accuracy": accuracy}

    def predict(self, X):
        if self.weights is None:
            raise Exception("Not trained yet !")
        return maths.sigmoid(np.dot(X, self.weights) + self.bias)

    def save(self, model_path):
        params = {
            "inits": {
                "alpha": self.alpha
            },
            "attrs": {
                "weights": self.weights,
                "bias": self.bias,
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
        y = df[4]
    X = df.drop(columns=[4]).values

    return X, y


def main(_):
    model_path = "./models/logistic_regression.pkl"

    if FLAGS.do_train:
        data_path = "../datasets/banknote_auth/train.csv"
        X_train, y_train = prepare_data(data_path, is_training=True)

        model = LogisticRegression(alpha=0.1)
        model.fit(X_train, y_train, epochs=20, batch_size=32)
        model.save(model_path)

    if FLAGS.do_eval:
        data_path = "../datasets/banknote_auth/dev.csv"
        X_dev, y_dev = prepare_data(data_path, is_training=False)

        model = LogisticRegression.restore(model_path)
        result = model.evaluate(X_dev, y_dev)
        print(result)


if __name__ == "__main__":
    app.run(main)
