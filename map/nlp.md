### Text Processing
- **Cleaning** : clean irrelevant contents such as HTML tags.
- **Normalization** : translate to lower case, deal with punctuation characters.
- **Tokenization** :  chop up into pieces, called tokens.
- **Stop word removal** : remove stop words which are uninformative like 'is', 'the', etc.
- **Lemmatization** : reduce a word to its normalized form.
- **Stemming** : reduce a word to its stem or root form.

### Part-of-speech tagging
*Part-of-speech tagging* is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context—i.e., its relationship with adjacent and related words in a phrase, sentence, or paragraph. It is often done by using *Hidden Markov Model (HMM)* along with *n-gram*.

### Hidden Markov Models (HMM)
*HMM* is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (i.e. hidden) states.

### n-gram
An *n-gram* is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. Using Latin numerical prefixes, an n-gram of size 1 is referred to as a "unigram"; size 2 is a "bigram" (or, less commonly, a "digram"); size 3 is a "trigram".

### Feature Extraction

#### Bag of Words
A *bag-of-words* is a representation of text that describes the occurrence of words within a document. It involves two things: *a vocabulary of known words* and *a measure of the presence of known words*. It is called a “bag” of words, because any information about the order or structure of words in the document is discarded.

#### TF-IDF
*TF-IDF* short for *term frequency–inverse document frequency*, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is the product of two statistics, *term frequency* and *inverse document frequency* which helps to adjust for the fact that some words appear more frequently in general.

#### Word Embedding
*Word embedding* is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers.

- **Word2Vec** : a group of related models that are used to produce word embeddings. It transforms words to vectors.
  - **Continuous Bag of Words (CboW)** : the model predicts the current word from a window of surrounding context words. The order of context words does not influence prediction (bag-of-words assumption).
  - **Skip-gram** : the model uses the current word to predict the surrounding window of context words. It weighs nearby context words more heavily than more distant context words.

- **GloVe** : coined from Global Vectors, is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

### Latent Dirichlet Allocation (LDA)
*LDA* is a generative probabilistic model of a corpus. The basic idea is
that documents are represented as random mixtures over latent topics, where each topic is characterized by a distribution over words.

### RNN
see [RNN](./deeplearning.md)
