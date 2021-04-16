# coding=utf-8

"""Tokenization classes."""

import collections
import json
import re
import unicodedata
from collections import Counter
from utils import learn_bpe


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    raise ValueError("Unsupported string type: %s" % (type(text)))


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    with open(vocab_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    vocab = collections.OrderedDict()
    reserved_tokens = []

    for i, item in enumerate(items):
        token = item
        if item.startswith("RESERVED:"):
            token = item.split(": ", 1)[1]
            reserved_tokens.append(token)
        vocab[token] = i

    return vocab, reserved_tokens


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _trans_token(token, vocab):
    """Replace characters that aren't in the vocab with its bytes unicode
    and appends a space " " to mark the end of a token.

    Args:
        token: unicode string to be transformed
        alphabet: list of all known characters

    Returns:
        transformed string
    """
    ret = []
    for char in token:
        if char in vocab:
            ret.append(char)
        else:
            ret.extend([chr(i) for i in char.encode("utf-8")])

    ret = "".join(ret)
    if len(ret) == 1 and (_is_chinese_char(ret)
                          or _is_punctuation_or_symbol(ret)):
        return ret
    return "".join(ret) + " "


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 max_token_len=100,
                 max_subtoken_len=20):
        self.vocab, self.reserved_tokens = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(
            reserved_tokens=self.reserved_tokens, do_lower_case=do_lower_case)
        self.subword_tokenizer = SubwordTokenizer(self.vocab, max_subtoken_len)
        self.max_token_len = max_token_len

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            if token in self.reserved_tokens:
                split_tokens.append(token)
                continue
            token = token[:self.max_token_len]
            token = _trans_token(token, self.vocab)
            for subtoken in self.subword_tokenizer.sub_tokenize(token):
                split_tokens.append(subtoken)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    @staticmethod
    def tokens2text(tokens):
        text = ""
        cur_ints = []
        for x in "".join(tokens):
            cp = ord(x)
            if cp >= 128 and cp < 256:
                cur_ints.append(cp)
                continue
            if cur_ints:
                try:
                    text += bytes(cur_ints).decode("utf-8")
                except UnicodeDecodeError as e:
                    text += "".join(map(chr, cur_ints))
                cur_ints = []
            text += x
        return text

    @classmethod
    def build_from_corpus(
            cls,
            corpus,
            reserved_tokens=None,
            target_vocab_size=2**15,
            min_frequency=2,
            vocab_file="vocab.json",
            do_lower_case=True,
            max_token_len=100,
            max_subtoken_len=20):
        if reserved_tokens is None:
            reserved_tokens = []
        basic_tokenizer = BasicTokenizer(reserved_tokens, do_lower_case)

        if type(corpus) == str:
            corpus_file = corpus
            corpus = []
            with open(corpus_file, encoding="utf-8") as f:
                for line in f:
                    corpus.append(line)

        alphabet = set(chr(i) for i in range(256))
        vocab_freq = Counter()
        for line in corpus:
            tokens = basic_tokenizer.tokenize(line)
            alphabet |= set("".join(tokens))
            vocab_freq.update(tokens)

        for token in reserved_tokens:
            del vocab_freq[token]

        for token in alphabet:
            if _is_chinese_char(token) or _is_punctuation_or_symbol(token):
                del vocab_freq[token]

        num_symbols = target_vocab_size - len(reserved_tokens) - len(alphabet)
        num_symbols = max([0, num_symbols])
        symbols = learn_bpe.learn_bpe(vocab_freq, num_symbols, min_frequency)

        token_end_pattern = re.compile(r"</w>$")
        vocab = reserved_tokens + sorted(alphabet)
        for symbol in symbols:
            symbol = token_end_pattern.sub(" ", symbol)
            if (symbol in reserved_tokens or symbol in alphabet or
                    len(symbol) > max_subtoken_len):
                continue
            vocab.append(symbol)

        cls._save_vocab(vocab_file, vocab, reserved_tokens)
        return cls(vocab_file, do_lower_case, max_token_len, max_subtoken_len)

    def save_vocab(self, vocab_file):
        self._save_vocab(vocab_file, self.vocab, self.reserved_tokens)

    @staticmethod
    def _save_vocab(vocab_file, vocab, reserved_tokens):
        if isinstance(vocab, dict):
            vocab = [t for t, _ in sorted(vocab.items(), key=lambda x: x[1])]

        items = []
        for token in vocab:
            item = ""
            if token in reserved_tokens:
                item += "RESERVED: "
            item += token
            items.append(item)

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, reserved_tokens=None, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        if reserved_tokens is None:
            reserved_tokens = []
        self.reserved_tokens = reserved_tokens

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token in self.reserved_tokens:
                split_tokens.append(token)
                continue
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc_or_symb(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc_or_symb(self, text):
        """Splits punctuation or symbol on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation_or_symbol(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            if _is_chinese_char(char):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class SubwordTokenizer(object):
    """Runs subword tokenziation."""

    def __init__(self, vocab, max_subtoken_len=20):
        self.vocab = vocab
        self.max_subtoken_len = max_subtoken_len

    def sub_tokenize(self, token):
        """Tokenizes a single word/token into its sub words/tokens .

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        Args:
            token: A single token. This should have already been passed
                   through `BasicTokenizer`.

        Returns:
            A list of sub tokens.
        """

        start = 0
        subtokens = []
        token_len = len(token)
        while start < token_len:
            end = min(token_len, start + self.max_subtoken_len)
            cur_substr = None
            while start < end:
                substr = token[start:end]
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1

            if cur_substr is None:
                # If there is no possible to sub tokenize the token, then one
                # of the characters in the token is not in the alphabet. This
                # should be impossible and would be indicative of a bug.
                raise ValueError(
                    "Unable to split token \"%s\" into subtokens." % token)
            subtokens.append(cur_substr)
            start = end

        return subtokens


class SimpleTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="<unk>"):
        self.vocab, self.reserved_tokens = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(
            reserved_tokens=self.reserved_tokens, do_lower_case=do_lower_case)
        self.unk_token = unk_token

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            if token not in self.vocab:
                token = self.unk_token
            split_tokens.append(token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    @staticmethod
    def tokens2text(tokens):
        return " ".join(tokens)

    @classmethod
    def build_from_corpus(
            cls,
            corpus,
            reserved_tokens=None,
            target_vocab_size=2**15,
            min_frequency=1,
            vocab_file="vocab.json",
            do_lower_case=True,
            unk_token="<unk>"):

        if reserved_tokens is None:
            reserved_tokens = [unk_token]
        if unk_token not in reserved_tokens:
            reserved_tokens.append(unk_token)

        basic_tokenizer = BasicTokenizer(reserved_tokens, do_lower_case)

        if type(corpus) == str:
            corpus_file = corpus
            corpus = []
            with open(corpus_file, encoding="utf-8") as f:
                for line in f:
                    corpus.append(line)

        vocab_freq = Counter()
        for line in corpus:
            tokens = basic_tokenizer.tokenize(line)
            vocab_freq.update(tokens)

        for token in reserved_tokens:
            del vocab_freq[token]

        num_symbols = target_vocab_size - len(reserved_tokens)
        vocab_freq = filter(lambda x: x[1] >= min_frequency, vocab_freq.items())
        vocab_freq = sorted(vocab_freq, key=lambda x: -x[1])[:num_symbols]
        vocab = reserved_tokens + list(map(lambda x: x[0], vocab_freq))

        cls._save_vocab(vocab_file, vocab, reserved_tokens)
        return cls(vocab_file, do_lower_case, unk_token=unk_token)

    def save_vocab(self, vocab_file):
        self._save_vocab(vocab_file, self.vocab, self.reserved_tokens)

    @staticmethod
    def _save_vocab(vocab_file, vocab, reserved_tokens):
        if isinstance(vocab, dict):
            vocab = [t for t, _ in sorted(vocab.items(), key=lambda x: x[1])]

        items = []
        for token in vocab:
            item = ""
            if token in reserved_tokens:
                item += "RESERVED: "
            item += token
            items.append(item)

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)


def _is_chinese_char(char):
    """Checks whether `char` is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode
    # block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean
    # characters, despite its name. The modern Korean Hangul alphabet is a
    # different block, as is Japanese Hiragana and Katakana. Those
    # alphabets are used to write space-separated words, so they are not
    # treated specially and handled like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or    #
            (cp >= 0x3040 and cp <= 0x30FF) or    #
            (cp >= 0x3400 and cp <= 0x4DBF) or    #
            (cp >= 0x20000 and cp <= 0x2A6DF) or    #
            (cp >= 0x2A700 and cp <= 0x2B73F) or    #
            (cp >= 0x2B740 and cp <= 0x2B81F) or    #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or    #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):    #
        return True

    return False


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation_or_symbol(char):
    """Checks whether `char` is a punctuation or symbol character."""
    cat = unicodedata.category(char)
    if cat.startswith("P") or cat.startswith("S"):
        return True
    return False
