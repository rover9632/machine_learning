"""Use byte pair encoding (BPE) to learn a variable-length encoding of the
vocabulary in a text. Unlike the original BPE, it does not compress the plain
text, but can be used to reduce the vocabulary of a text to a configurable
number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine
Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational
Linguistics (ACL 2016). Berlin, Germany.
"""

import re
import copy
from collections import defaultdict, Counter
import time


def get_vocabulary(fobj, is_dict=False):
    """Read text and return dictionary that encodes vocabulary
    """
    vocab = Counter()
    for i, line in enumerate(fobj):
        if is_dict:
            try:
                word, count = line.strip('\r\n ').split(' ')
            except Exception:
                raise Exception(
                    'Failed reading vocabulary file at line {0}: {1}'.format(
                        i, line))
            vocab[word] += int(count)
        else:
            for word in line.strip('\r\n ').split(' '):
                if word:
                    vocab[word] += 1
    return vocab


def update_pair_statistics(pair, changed, stats, indices):
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first + second
    for j, word, old_word, freq in changed:

        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an
            # occurrence of pair (old_word[i:i+2])
            if i < len(old_word) - 1 and old_word[i + 1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged,
                # reduce the frequency of "A B"
                if i:
                    prev = old_word[i - 1:i + 1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word) - 2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged,
                    # reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because
                    # the frequency of "C B" will be reduced by the previous
                    # code block
                    if old_word[i + 2] != first or i >= len(
                            old_word) - 3 or old_word[i + 3] != second:
                        nex = old_word[i + 1:i + 3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged,
            # increase the frequency of "A BC"
            if i:
                prev = word[i - 1:i + 1]
                stats[prev] += freq
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase
            # the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the
            # count of "BC BC" will be incremented by the previous code block
            if i < len(word) - 1 and word[i + 1] != new_pair:
                nex = word[i:i + 2]
                stats[nex] += freq
                indices[nex][j] += 1
            i += 1


def get_pair_statistics(vocab):
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int)

    # index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices):
    """
    Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'
    """
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\', '\\\\')
    changes = []
    ptn = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')

    for j, freq in indices[pair].items():
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = ptn.sub(pair_str, new_word)
        new_word = tuple(new_word.split(' '))

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes


def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we
    previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item, freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def learn_bpe(vocab_freq, num_symbols, min_frequency=2):
    """Learn num_symbols BPE operations from vocabulary, and return the symbols.
    """

    vocab = dict([(tuple(x) + ('</w>', ), y) for (x, y) in vocab_freq.items()])
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_statistics(sorted_vocab)
    big_stats = copy.deepcopy(stats)

    # threshold is inspired by Zipfian assumption, but should only affect speed
    threshold = max(stats.values()) / 10
    symbols = []
    for i in range(num_symbols):
        if stats:
            most_frequent = max(stats, key=lambda x: (stats[x], x))

        # we probably missed the best pair because of pruning; go back to full
        # statistics
        if not stats or (i and stats[most_frequent] < threshold):
            prune_stats(stats, big_stats, threshold)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x], x))
            # threshold is inspired by Zipfian assumption, but should only
            # affect speed
            threshold = stats[most_frequent] * i / (i + 10000.0)
            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent] < min_frequency:
            break

        symbols.append("".join(most_frequent))
        changes = replace_pair(most_frequent, sorted_vocab, indices)
        update_pair_statistics(most_frequent, changes, stats, indices)
        stats[most_frequent] = 0
        if not i % 100:
            prune_stats(stats, big_stats, threshold)

    return symbols


def main(args):
    start = time.time()

    inp = open(args[0], encoding="utf-8")
    vocab_freq = get_vocabulary(inp)
    symbols = learn_bpe(vocab_freq, 1000)
    with open(args[1], "w", encoding="utf-8") as f:
        for symbol in symbols:
            f.write("'%s'\n" % symbol)

    end = time.time()
    print("Time cost: %d seconds, about %d minutes" %
          (int(end - start), int((end - start) / 60)))
