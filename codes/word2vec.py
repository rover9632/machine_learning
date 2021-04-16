from os import path
from collections import Counter, defaultdict

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import tokenization

flags.DEFINE_bool("do_train", default=False, help="do training the model")
flags.DEFINE_bool("do_resume",
                  default=False,
                  help="resume latest checkpoint and continue train the model")
flags.DEFINE_bool("do_eval", default=False, help="do evaluation")
flags.DEFINE_bool("do_predict", default=False, help="do prediction")

FLAGS = flags.FLAGS
_EPSILON = 1e-7


class Word2Vec(keras.Model):

    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.input_embedding = layers.Embedding(vocab_size,
                                                emb_dim,
                                                input_length=1,
                                                name="w2v_embedding")
        self.output_embedding = layers.Embedding(vocab_size, emb_dim)
        self.flatten = layers.Flatten()

    def call(self, inputs):
        v_in = self.input_embedding(inputs[0])
        v_out = self.output_embedding(inputs[1])
        return self.flatten(tf.matmul(v_in, v_out, transpose_b=True))

    def words_to_vectors(self, words):
        words = tf.convert_to_tensor(words, dtype=tf.int32)
        return self.input_embedding(words)

    def similarity(self, word1, word2):
        words = tf.convert_to_tensor([word1, word2], dtype=tf.int32)
        v1, v2 = tf.math.l2_normalize(self.input_embedding(words), axis=-1)
        sim = tf.matmul(tf.expand_dims(v1, -2), tf.expand_dims(v2, -1))
        return tf.squeeze(sim)

    def sentence_similarity(self, sen1, sen2):
        sen1 = tf.convert_to_tensor(sen1, dtype=tf.int32)
        sen2 = tf.convert_to_tensor(sen2, dtype=tf.int32)
        mask1 = tf.expand_dims(tf.cast(sen1 != 0, dtype=tf.float32), axis=-1)
        mask2 = tf.expand_dims(tf.cast(sen2 != 0, dtype=tf.float32), axis=-1)
        sen1 = self.input_embedding(sen1) * mask1
        sen2 = self.input_embedding(sen2) * mask2
        sen1 = tf.math.reduce_sum(sen1, axis=-2, keepdims=True)
        sen2 = tf.math.reduce_sum(sen2, axis=-2, keepdims=True)
        sen1 = tf.math.l2_normalize(sen1, axis=-1)
        sen2 = tf.math.l2_normalize(sen2, axis=-1)
        return tf.squeeze(tf.matmul(sen1, sen2, transpose_b=True))

    def pearson(self, sens1, sens2, labels):
        preds = self.sentence_similarity(sens1, sens2)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        avg_pred = tf.math.reduce_mean(preds)
        avg_label = tf.math.reduce_mean(labels)
        prsn = tf.math.reduce_mean((preds - avg_pred) * (labels - avg_label))
        prsn = prsn / (tf.math.reduce_std(preds) * tf.math.reduce_std(labels))
        return prsn


class NegativeSamplingLoss(keras.losses.Loss):

    def __init__(self, name="negative_sampling_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        probs = tf.math.sigmoid(y_pred)
        loss = -y_true * tf.math.log(probs + _EPSILON)
        loss -= (1.0 - y_true) * tf.math.log(1.0 - probs + _EPSILON)
        return tf.math.reduce_mean(loss, axis=-1)


class Accuracy(keras.metrics.Metric):

    def __init__(self, name="accuracy"):
        super().__init__(name=name)
        self.corrects = self.add_weight(name="corrects", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        probs = tf.math.sigmoid(y_pred)
        preds = tf.cast(probs >= 0.5, dtype=y_true.dtype)
        corrs = tf.cast(y_true == preds, dtype=tf.float32)
        self.corrects.assign_add(tf.math.reduce_sum(corrs))
        self.total.assign_add(tf.cast(tf.size(y_true), dtype=tf.float32))

    def result(self):
        return self.corrects / (self.total + _EPSILON)

    def reset_states(self):
        self.corrects.assign(0.0)
        self.total.assign(0.0)


class NegativeSampler():

    def __init__(self, word_freq):
        for k, v in word_freq.items():
            word_freq[k] = np.math.pow(v, 0.75)

        words, freqs = zip(*word_freq.items())
        self.words = np.array(words)
        self.freqs = np.array(freqs) / np.sum(freqs)

    def negative_sample(self, num_words):
        return np.random.choice(self.words, size=num_words, p=self.freqs)


class Subsampler():

    def __init__(self, word_freq, threshold=1e-5):
        total = np.sum(list(word_freq.values()))
        self.discard_probs = defaultdict(float)

        for k, v in word_freq.items():
            prob = 1.0 - np.math.sqrt(threshold * total / v)
            self.discard_probs[k] = max(prob, 0.0)

    def subsample(self, data):
        new_data = []
        for x in data:
            prob = self.discard_probs[x[0]]
            if np.random.choice([True, False], p=[prob, 1.0 - prob]):
                continue
            new_data.append(x)
        return new_data


def get_word_freq(corpus, tokenizer):
    word_freq = Counter()
    for line in corpus:
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
        word_freq.update(tokens)
    return word_freq


def prepare_training_data(tokenizer,
                          data_path,
                          window=5,
                          num_negs=5,
                          batch_size=32):
    corpus = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            corpus.append(line.rstrip())
            '''
            s1, s2, _ = line.split("\t")
            corpus.append(s1)
            corpus.append(s2)
            '''

    word_pairs = []
    for line in corpus:
        words = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
        for i in range(len(words)):
            context = words[max(0, i - window):i] + words[i + 1:i + window + 1]
            word_pairs.extend(map(lambda x: (words[i], x), context))

    word_freq = get_word_freq(corpus, tokenizer)
    subsampler = Subsampler(word_freq, 1e-5)
    word_pairs = subsampler.subsample(word_pairs)

    neg_sampler = NegativeSampler(word_freq)
    negs_all = neg_sampler.negative_sample(len(word_pairs) * num_negs)
    negs_all = np.reshape(negs_all, (len(word_pairs), num_negs))
    targets = []
    contexts = []
    labels = []
    for (t, c), negs in zip(word_pairs, negs_all):
        neg_labels = [0.0] * len(negs)
        targets.append([t])
        contexts.append([c] + list(negs))
        labels.append([1.0] + neg_labels)

    ds = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    ds = ds.shuffle(100000).batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def prepare_eval_data(tokenizer, data_path, seq_len=64):
    sens1 = []
    sens2 = []
    labels = []

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            s1, s2, label = line.split("\t")
            s1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s1))
            s2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s2))
            sens1.append(pad_or_truncate_seq(s1, seq_len))
            sens2.append(pad_or_truncate_seq(s2, seq_len))
            labels.append(float(label))

    return sens1, sens2, labels


def pad_or_truncate_seq(seq, seq_len, pad_val=0):
    if len(seq) < seq_len:
        seq.extend([pad_val] * (seq_len - len(seq)))
    elif len(seq) > seq_len:
        seq = seq[:seq_len]
    return seq


def build_tokenizer(data_path, vocab_file):
    corpus = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            corpus.append(line.rstrip())

    tokenizer = tokenization.SimpleTokenizer.build_from_corpus(
        corpus=corpus,
        reserved_tokens=["<pad>", "<unk>"],
        target_vocab_size=2**15,
        min_frequency=2,
        vocab_file=vocab_file,
        do_lower_case=True,
        unk_token="<unk>")

    return tokenizer


def main(_):
    WINDOW = 5
    NUM_NEGS = 5
    EMB_DIM = 300
    BATCH_SIZE = 32
    EPOCHS = 10

    model_dir = "./models/word2vec"
    vocab_file = path.join(model_dir, "vocab.json")

    if FLAGS.do_train:
        data_path = "../datasets/stsbenchmark/exdata.txt"
        tokenizer = build_tokenizer(data_path, vocab_file)

        ds_train = prepare_training_data(tokenizer,
                                         data_path,
                                         window=WINDOW,
                                         num_negs=NUM_NEGS,
                                         batch_size=BATCH_SIZE)

        model = Word2Vec(len(tokenizer.vocab), EMB_DIM)
        model.compile(optimizer=keras.optimizers.Adam(0.001),
                      loss=NegativeSamplingLoss(),
                      metrics=[Accuracy()])

        ckpt_path = path.join(model_dir, "checkpoints", "ckpt-{epoch}")
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                        save_weights_only=True)
        model.fit(ds_train, epochs=EPOCHS, callbacks=[ckpt_callback])

    if FLAGS.do_eval:
        data_path = "../datasets/stsbenchmark/dev.tsv"
        tokenizer = tokenization.SimpleTokenizer(vocab_file)
        sens1, sens2, labels = prepare_eval_data(tokenizer, data_path)

        model = Word2Vec(len(tokenizer.vocab), EMB_DIM)
        status = model.load_weights(
            tf.train.latest_checkpoint(path.join(model_dir, "checkpoints")))
        status.expect_partial()

        result = model.pearson(sens1, sens2, labels).numpy()
        print({"pearson": result})


if __name__ == "__main__":
    app.run(main)
