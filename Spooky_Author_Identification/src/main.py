# coding: utf-8

# Load and check data

from tqdm import tqdm
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import gensim
import nltk

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics, model_selection
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from keras.models import Sequential, model_from_json
from keras.layers import SpatialDropout1D, Bidirectional, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

import xgboost as xgb

# Load in the train and test datasets
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

data_full = pd.concat([data_train, data_test], ignore_index=True)

print(data_train.shape, data_test.shape, data_full.shape)

data_full.head()

# Prepare the id and label for modeling
author_mapping_dict = {'EAP': 0, 'HPL': 1, 'MWS': 2}

#y_full = data_full.author
y_full = data_full.author.map(author_mapping_dict)

i_train = ~y_full.isnull()
i_test = y_full.isnull()

y_train = data_train.author.map(author_mapping_dict)
#y_train = y_full[i_train]

# Explore the dataset
print("Number of rows in train dataset {}".format(data_train.shape[0]))
print("Number of rows in test dataset {}".format(data_test.shape[0]))
print("No duplicates in train data") if data_train.shape[
    0] == data_train.text.unique().__len__() else print("Oops")
print("No duplicates in test data") if data_test.shape[
    0] == data_test.text.unique().__len__() else print("Oops")
print("Unique author is data {}".format(data_train.author.unique()))
print("Number of nulls in the train is {} and text is {}".format(
    data_train.isnull().sum().sum(), data_test.isnull().sum().sum()))

# # Natural Language Processing

# ## Tokenization

# Storing the first text element as a string
first_text = data_train.text.values[0]
print(first_text)
print("=" * 90)
print(first_text.split(" "))

first_text_list = nltk.word_tokenize(first_text)
print(first_text_list)

# ## Stopword Removal

stopwords = nltk.corpus.stopwords.words('english')
len(stopwords)

first_text_list_cleaned = [
    word for word in first_text_list if word.lower() not in stopwords
]
print(first_text_list_cleaned)
print("=" * 90)
print("Length of original list: {} words\n"
      "Length of list after stopwords removal: {} words".format(
          len(first_text_list), len(first_text_list_cleaned)))

# ## Stemming and Lemmatization

stemmer = nltk.stem.PorterStemmer()
print("The stemmed form of running is: {}".format(stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(stemmer.stem("run")))

print("The stemmed form of leaves is: {}".format(stemmer.stem("leaves")))

lemm = nltk.stem.WordNetLemmatizer()
print("The lemmatized form of leaves is: {}".format(lemm.lemmatize("leaves")))
print("The lemmatized form of leaves is: {}".format(
    lemm.lemmatize("ascertaining")))

# ## Vectorizing Raw Text

# Defining our sentence
sentence = ["I love to eat Burgers", "I love to eat Fries"]
# try CountVectorizer
vectorizer = CountVectorizer(min_df=0)
sentence_transform = vectorizer.fit_transform(sentence)

print("The features are:\n {}".format(vectorizer.get_feature_names()))
print("\nThe vectorized array looks like:\n {}".format(
    sentence_transform.toarray()))

sentence = ["I love to eat Burgers", "I love to eat Fries"]
# try  TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0)
sentence_transform = vectorizer.fit_transform(sentence)

print("The features are:\n {}".format(vectorizer.get_feature_names()))
print("\nThe vectorized array looks like:\n {}".format(
    sentence_transform.toarray()))

# # Feature Engineering

# ## Basic features

import re
import string


def num_words(raw):
    return len(re.findall(r'\w+', raw['text']))


def num_chars(raw):
    return len(raw['text'])


def mean_len_words(raw):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(raw['text'])

    return np.mean([len(w) for w in text_list])


def num_unique_words(raw):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(raw['text'].lower())

    return len(list(set(text_list)))


def num_stopwords(raw):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(raw['text'].lower())

    stopwords = nltk.corpus.stopwords.words('english')

    return len([w for w in text_list if w in stopwords])


def num_punctuations(raw):
    text = raw['text'].lower()
    text_list = nltk.word_tokenize(text)

    return len([w for w in text_list if w in string.punctuation])


def num_upper_words(raw):
    text = raw['text']
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    return len([w for w in text_list if w.isupper()])


def num_title_words(raw):
    text = raw['text']
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    return len([w for w in text_list if w.istitle()])


def num_noun_words(raw):
    text = raw['text'].lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    pos_list = nltk.pos_tag(text_list)
    noun_count = len(
        [w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    return noun_count


def num_adj_words(raw):
    text = raw['text'].lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    pos_list = nltk.pos_tag(text_list)
    adj_count = len([w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')])
    return adj_count


def num_verbs_words(raw):
    text = raw['text'].lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    pos_list = nltk.pos_tag(text_list)
    verbs_count = len([
        w for w in pos_list
        if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    ])
    return verbs_count


def fra_unique_words(raw):
    text = raw['text'].lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    unique_word_count = len(list(set(text_list)))
    word_count = len(text_list)
    return unique_word_count / word_count


def fra_stopwords(raw):
    text = raw['text'].lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    stopwords = nltk.corpus.stopwords.words('english')

    stopwords_count = len([w for w in text_list if w in stopwords])
    word_count = len(text_list)
    return stopwords_count / word_count


def fra_punctuations(raw):
    text = raw['text'].lower()
    text_list = nltk.word_tokenize(text)

    punctuation_count = len([w for w in text_list if w in string.punctuation])
    char_count = len(text)
    return punctuation_count / char_count


def fra_upper_words(raw):
    text = raw['text']
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    word_upper_count = len([w for w in text_list if w.isupper()])
    word_count = len(text_list)
    return word_upper_count / word_count


def fra_title_words(raw):
    text = raw['text']
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    word_title_count = len([w for w in text_list if w.istitle()])
    word_count = len(text_list)
    return word_title_count / word_count


def fra_noun_words(raw):
    text = raw['text'].lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    pos_list = nltk.pos_tag(text_list)
    noun_count = len(
        [w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    word_count = len(text_list)
    return noun_count / word_count


def fra_adj_words(raw):
    text = raw['text'].lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    pos_list = nltk.pos_tag(text_list)
    adj_count = len([w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')])
    word_count = len(text_list)
    return adj_count / word_count


def fra_verbs_words(raw):
    text = raw['text'].lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)

    pos_list = nltk.pos_tag(text_list)
    verbs_count = len([
        w for w in pos_list
        if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    ])
    word_count = len(text_list)
    return verbs_count / word_count


# In[23]:

f_num_words = data_full.apply(
    lambda raw: num_words(raw), axis=1).as_matrix()[:, None]
f_num_chars = data_full.apply(
    lambda raw: num_chars(raw), axis=1).as_matrix()[:, None]
f_mean_len_words = data_full.apply(
    lambda raw: mean_len_words(raw), axis=1).as_matrix()[:, None]

f_num_unique_words = data_full.apply(
    lambda raw: num_unique_words(raw), axis=1).as_matrix()[:, None]
f_num_stopwords = data_full.apply(
    lambda raw: num_stopwords(raw), axis=1).as_matrix()[:, None]
f_num_punctuations = data_full.apply(
    lambda raw: num_punctuations(raw), axis=1).as_matrix()[:, None]
f_num_upper_words = data_full.apply(
    lambda raw: num_upper_words(raw), axis=1).as_matrix()[:, None]
f_num_title_words = data_full.apply(
    lambda raw: num_title_words(raw), axis=1).as_matrix()[:, None]

f_num_noun_words = data_full.apply(
    lambda raw: num_noun_words(raw), axis=1).as_matrix()[:, None]
f_num_adj_words = data_full.apply(
    lambda raw: num_adj_words(raw), axis=1).as_matrix()[:, None]
f_num_verbs_words = data_full.apply(
    lambda raw: num_verbs_words(raw), axis=1).as_matrix()[:, None]

f_fra_unique_words = data_full.apply(
    lambda raw: fra_unique_words(raw), axis=1).as_matrix()[:, None]
f_fra_stopwords = data_full.apply(
    lambda raw: fra_stopwords(raw), axis=1).as_matrix()[:, None]
f_fra_punctuations = data_full.apply(
    lambda raw: fra_punctuations(raw), axis=1).as_matrix()[:, None]
f_fra_upper_words = data_full.apply(
    lambda raw: fra_upper_words(raw), axis=1).as_matrix()[:, None]
f_fra_title_words = data_full.apply(
    lambda raw: fra_title_words(raw), axis=1).as_matrix()[:, None]

f_fra_noun_words = data_full.apply(
    lambda raw: fra_noun_words(raw), axis=1).as_matrix()[:, None]
f_fra_adj_words = data_full.apply(
    lambda raw: fra_adj_words(raw), axis=1).as_matrix()[:, None]
f_fra_verbs_words = data_full.apply(
    lambda raw: fra_verbs_words(raw), axis=1).as_matrix()[:, None]

f_basic = (f_num_words, f_num_chars, f_mean_len_words, f_num_unique_words,
           f_num_stopwords, f_num_punctuations, f_num_title_words,
           f_num_upper_words, f_num_noun_words, f_num_adj_words,
           f_num_verbs_words, f_fra_unique_words, f_fra_stopwords,
           f_fra_punctuations, f_fra_title_words, f_fra_upper_words,
           f_fra_noun_words, f_fra_adj_words, f_fra_verbs_words)

# ## Word features

# ### Word based 6 features

n_components = 40

# TFIDF features
t_tfidf = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 3))  # token_pattern=r'\w{1,}'
f_tfidf = t_tfidf.fit_transform(data_full.text)

# TFIDF with SVD features
t_svd_tfidf = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_tfidf = t_svd_tfidf.fit_transform(f_tfidf)

# TFIDF with SVD and Scale features
t_scl_svd_tfidf = StandardScaler()
f_scl_svd_tfidf = t_scl_svd_tfidf.fit_transform(f_svd_tfidf)

# Counters features
# min_df=3, strip_accents='unicode', token_pattern=r'\w{1,}', sublinear_tf=1,
t_count = CountVectorizer(stop_words="english", ngram_range=(1, 3))
f_count = t_count.fit_transform(data_full.text)

# Counters with SVD features
t_svd_count = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_count = t_svd_count.fit_transform(f_count.astype(float))  # float

# Counter with SVD and Scale features
t_scl_svd_count = StandardScaler()
f_scl_svd_count = t_scl_svd_count.fit_transform(f_svd_count)

# ### Stems based 6 features

# In[27]:

stemmer = nltk.stem.PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize_s(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = stem_tokens(tokens, stemmer)
    return stems


# In[28]:

# TFIDF features
t_tfidf_s = TfidfVectorizer(
    tokenizer=tokenize_s, stop_words="english", ngram_range=(1, 3))
f_tfidf_s = t_tfidf_s.fit_transform(data_full.text)

# TFIDF with SVD features
t_svd_tfidf_s = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_tfidf_s = t_svd_tfidf_s.fit_transform(f_tfidf_s)

# TFIDF with SVD and Scale features
t_scl_svd_tfidf_s = StandardScaler()
f_scl_svd_tfidf_s = t_scl_svd_tfidf_s.fit_transform(f_svd_tfidf_s)

# Counters features
t_count_s = CountVectorizer(
    tokenizer=tokenize_s, stop_words="english", ngram_range=(1, 3))
f_count_s = t_count_s.fit_transform(data_full.text)

# Counters with SVD features
t_svd_count_s = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_count_s = t_svd_count_s.fit_transform(f_count_s.astype(float))

# Counters with SVD and Scale features
t_scl_svd_count_s = StandardScaler()
f_scl_svd_count_s = t_scl_svd_count_s.fit_transform(f_svd_count_s)

# ### Lemmas based 6 features

# In[29]:

lemmatizer = nltk.stem.WordNetLemmatizer()


def tokenize_l(text):
    lemms = []
    for i, j in nltk.pos_tag(nltk.word_tokenize(text.lower())):
        if j[0].lower() in ['a', 'n', 'v']:
            lemms.append(lemmatizer.lemmatize(i, j[0].lower()))
        else:
            lemms.append(lemmatizer.lemmatize(i))
    return lemms


# In[30]:

# TFIDF features
t_tfidf_l = TfidfVectorizer(
    tokenizer=tokenize_l, stop_words="english", ngram_range=(1, 3))
f_tfidf_l = t_tfidf_l.fit_transform(data_full.text)

# TFIDF with SVD features
t_svd_tfidf_l = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_tfidf_l = t_svd_tfidf_l.fit_transform(f_tfidf_l)

# TFIDF with SVD and Scale features
t_scl_svd_tfidf_l = StandardScaler()
f_scl_svd_tfidf_l = t_scl_svd_tfidf_l.fit_transform(f_svd_tfidf_l)

# Counters features
t_count_l = CountVectorizer(
    tokenizer=tokenize_l, stop_words="english", ngram_range=(1, 3))
f_count_l = t_count_l.fit_transform(data_full.text)

# Counters with SVD features
t_svd_count_l = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_count_l = t_svd_count_l.fit_transform(f_count_l.astype(float))

# Counter with SVD and Scale features
t_scl_svd_count_l = StandardScaler()
f_scl_svd_count_l = t_scl_svd_count_l.fit_transform(f_svd_count_l)

# ## Char 6 featurs

# In[31]:

# TFIDF features
t_tfidf_c = TfidfVectorizer(
    analyzer="char", stop_words="english", ngram_range=(1, 7))
f_tfidf_c = t_tfidf_c.fit_transform(data_full.text)

# TFIDF with SVD features
t_svd_tfidf_c = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_tfidf_c = t_svd_tfidf_c.fit_transform(f_tfidf_c)

# TFIDF with SVD and Scale features
t_scl_svd_tfidf_c = StandardScaler()
f_scl_svd_tfidf_c = t_scl_svd_tfidf_c.fit_transform(f_svd_tfidf_c)

# Counter features
t_count_c = CountVectorizer(
    analyzer="char", stop_words="english", ngram_range=(1, 7))
f_count_c = t_count_c.fit_transform(data_full.text)

# Counter with SVD features
t_svd_count_c = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_count_c = t_svd_count_c.fit_transform(f_count_c.astype(float))

# Counter with SVD and Scale features
t_scl_svd_count_c = StandardScaler()
f_scl_svd_count_c = t_scl_svd_count_c.fit_transform(f_svd_count_c)

# ## Markov event based features

# In[32]:


class Dictogram(dict):
    def __init__(self, iterable=None):
        """Initialize this histogram as a new dict; update with given items"""
        super(Dictogram, self).__init__()
        self.types = 0  # the number of distinct item types in this histogram
        self.tokens = 0  # the total count of all item tokens in this histogram
        if iterable:
            self.update(iterable)

    def update(self, iterable):
        """Update this histogram with the items in the given iterable"""
        for item in iterable:
            if item in self:
                self[item] += 1
                self.tokens += 1
            else:
                self[item] = 1
                self.types += 1
                self.tokens += 1


# In[33]:


# markov chain based features, order words memory
def make_higher_order_markov_model(data, order):
    markov_model = dict()

    for char_list in data:
        for i in range(len(char_list) - order):
            # Create the window
            window = tuple(char_list[i:i + order])
            # Add to the dictionary
            if window in markov_model:
                # We have to just append to the existing Dictogram
                markov_model[window].update([char_list[i + order]])
            else:
                markov_model[window] = Dictogram([char_list[i + order]])
    return markov_model


def make_tuples(char_list, order):
    """function to make tuples of order size given a char_list and order"""
    list_of_tuple = []
    chars = []
    for i in range(len(char_list) - order):
        window = tuple(char_list[i:i + order])
        list_of_tuple.append(window)
        chars.append(char_list[i + order])
    return (list_of_tuple, chars)


def sent_to_prob(raw, order, MM):
    """function to get the markov model to give prob of a author given a char_list """
    char_list = raw['splited_char_list']
    list_of_tuples, chars = make_tuples(char_list, order)

    p = 0

    # convert to log so we can sum probabilities instead of multiply
    for i in range(len(chars)):
        try:
            p_char = MM[list_of_tuples[i]][chars[i]]
            p_chars = sum([x for x in MM[list_of_tuples[i]].values()])
        except:
            p_char = 1
            p_chars = 1
        p += np.log(p_char / p_chars)
    return p


def text_to_char_list(raw):
    text = raw['text'].lower()
    char_list = [c for c in text]

    return char_list


# In[34]:

start_order, end_order = 2, 5  # [start_order, end_order)

data_full['splited_char_list'] = data_full.apply(
    lambda raw: text_to_char_list(raw), axis=1)

raw_eap = data_full[data_full.author == 'EAP']['splited_char_list'].values
raw_hpl = data_full[data_full.author == 'HPL']['splited_char_list'].values
raw_mws = data_full[data_full.author == 'MWS']['splited_char_list'].values

f_markov_all = []

for order in range(start_order, end_order):
    # build markov model
    eap_MM = make_higher_order_markov_model(raw_eap, order)
    hpl_MM = make_higher_order_markov_model(raw_hpl, order)
    mws_MM = make_higher_order_markov_model(raw_mws, order)

    f_markov_order = []
    # create markov features
    f_markov_order.append(
        data_full.apply(lambda raw: sent_to_prob(raw, order, eap_MM), axis=1)
        .as_matrix()[:, None])
    f_markov_order.append(
        data_full.apply(lambda raw: sent_to_prob(raw, order, hpl_MM), axis=1)
        .as_matrix()[:, None])
    f_markov_order.append(
        data_full.apply(lambda raw: sent_to_prob(raw, order, mws_MM), axis=1)
        .as_matrix()[:, None])

    f_markov_all.append(f_markov_order)

del data_full['splited_char_list']

# In[35]:

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

f_markov_all_t = np.array(f_markov_all)

f_markov_orders_t = []

for ii in range(end_order - start_order):
    f_markov_ii = np.hstack(f_markov_all_t[ii])
    f_markov_tt = np.zeros(f_markov_ii.shape)
    for jj, row in enumerate(f_markov_ii):
        f_markov_tt[jj] = (row - row.min()) / (row.max() - row.min())
    f_markov_tt2 = normalize(f_markov_ii)
    f_markov_tt3 = StandardScaler().fit_transform(f_markov_ii)

    f_markov_orders_t.append(
        np.hstack((f_markov_tt, f_markov_tt2, f_markov_tt3)))
f_markov_orders = np.hstack(f_markov_orders_t)

#print(eap_MM[('n', 'o', 'm', 'e')])
#print(sum([x for x in eap_MM[('n', 'o', 'm', 'e')].values()]))
#print(eap_MM[('n', 'o', 'm', 'e')]['r'])
#print(eap_MM[('n', 'o', 'm', 'e')]['r'] / sum([x for x in eap_MM[('n', 'o', 'm', 'e')].values()]))

# ## Sentence vector features

# ### word2vec

# Load Google's pre-trained Word2Vec model.
word2vec = gensim.models.KeyedVectors.load_word2vec_format(
    '../model/GoogleNews-vectors-negative300.bin', binary=True)


def sent2vec(sentence):
    stopwords = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    text_list = tokenizer.tokenize(sentence.lower())
    text_list = [w for w in text_list if w not in stopwords]
    text_list = [w for w in text_list if w.isalpha()]

    M = []
    for w in text_list:
        try:
            M.append(word2vec[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v**2).sum())


sent2v = [sent2vec(x) for x in data_full.text]
f_sent2v = np.array(sent2v)
print(f_sent2v[0])

#sve_train = np.load('../model/sve_train.npy')
#sve_test = np.load('../model/sve_test.npy')

#f_sent2v_t = np.concatenate((sve_train, sve_test), axis=0)
#print(f_sent2v_t[0])

# ### glove

glove_file = '../model/glove.840B.300d.txt'


# load the GloVe vectors in a dictionary:
def loadGloveEmbeddings(glove_file):
    embeddings_index = {}
    with open(glove_file) as f:
        for line in tqdm(f):
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


glove2vec = loadGloveEmbeddings(glove_file)

# ## Sentiment Analysis based features


def apply_nn_model(features,
                   model_func,
                   input_dim=None,
                   embedding_dims=None,
                   max_len=None,
                   embedding_matrix=None):
    #assert model_name in ('nn_model', 'sent2vec_model', 'fasttext_model', 'lstm_model', 'bi_lstm_model', 'gru_model')

    n_splits = 5
    kf = model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=2017)

    epochs = 100
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=2, verbose=0, mode='auto')

    x_train = features[np.nonzero(i_train)]
    x_test = features[np.nonzero(i_test)]
    y_train_enc = np_utils.to_categorical(y_train)

    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([x_train.shape[0], 3])

    for idx_dev, idx_val in kf.split(x_train):
        x_dev, x_val = x_train[idx_dev], x_train[idx_val]
        y_dev, y_val = y_train_enc[idx_dev], y_train_enc[idx_val]

        model = model_func(input_dim, embedding_dims, max_len,
                           embedding_matrix)

        #if model_name is 'nn_model':
        #    model = nn_model(input_dim, embedding_dims, max_len)
        #elif model_name is 'sent2vec_model':
        #    model = sent2vec_model()
        #elif model_name is 'fasttext_model':
        #    model = fasttext_model(input_dim, embedding_dims)
        #elif model_name is 'lstm_model':
        #    model = lstm_model(input_dim, embedding_dims, max_len,
        #                       embedding_matrix)
        #elif model_name is 'bi_lstm_model':
        #    model = bi_lstm_model(input_dim, embedding_dims, max_len,
        #                          embedding_matrix)
        #elif model_name is 'gru_model':
        #    model = gru_model(input_dim, embedding_dims, max_len,
        #                      embedding_matrix)

        model.fit(
            x_dev,
            y_dev,
            batch_size=32,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=[earlyStopping])
        pred_y_val = model.predict(x_val)
        pred_y_test = model.predict(x_test)
        pred_full_test = pred_full_test + pred_y_test
        pred_train[idx_val, :] = pred_y_val
        cv_scores.append(metrics.log_loss(y_val, pred_y_val))
    pred_full_test = pred_full_test / float(n_splits)
    print("Mean cv score: {}".format(np.mean(cv_scores)))

    p_full = np.concatenate((pred_train, pred_full_test), axis=0)
    return pd.DataFrame(p_full)


def nn_model(input_dim, embedding_dims, max_len, embedding_matrix):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=input_dim,
            output_dim=embedding_dims,
            input_length=max_len))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, 5, padding='valid', activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


#max_len = 35
max_len = 70
num_words = 10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(list(data_full.text))
f_nn_full_seq = tokenizer.texts_to_sequences(data_full.text)
f_nn_full_pad = pad_sequences(sequences=f_nn_full_seq, maxlen=max_len)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# nn model
input_dim = min(num_words, len(word_index)) + 1
embedding_dims = 32
max_len = 70
p_nn = apply_nn_model(f_nn_full_pad, nn_model, input_dim, embedding_dims,
                      max_len)

# Sent2vec Neural Network


def sent2vec_model(input_dim, embedding_dims, max_len, embedding_matrix):
    # create a simple 3 layer sequential neural net
    model = Sequential()

    model.add(Dense(128, input_dim=300))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


t_scl_sent2v_nn = StandardScaler()
f_scl_sent2v_nn = t_scl_sent2v_nn.fit_transform(f_sent2v)

# sent2vec nn model
p_sent2v_nn = apply_nn_model(f_scl_sent2v_nn, sent2vec_model)


# Separate punctuation from words
# Remove lower frequency words ( <= 2)
# Cut a longer document which contains 256 words
def preprocessFastText(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign))
    return text


def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
        ngrams = []
        for n in range(2, n_gram_max + 1):
            for w_index in range(len(q) - n + 1):
                ngrams.append('--'.join(q[w_index:w_index + n]))
        return q + ngrams

    docs = []
    for doc in df.text:
        doc = preprocessFastText(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))

    return docs


def fasttext_model(input_dim, embedding_dims, max_len, embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


min_count = 2
max_len = 90

docs = create_docs(data_full)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)

num_words = sum(
    [1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=True, filters='')
tokenizer.fit_on_texts(docs)
f_ft_docs_seq = tokenizer.texts_to_sequences(docs)
f_ft_docs_pad = pad_sequences(sequences=f_ft_docs_seq, maxlen=max_len)

# fasttext model
input_dim = np.max(f_ft_docs_pad) + 1
embedding_dims = 20
p_fasttext = apply_nn_model(f_ft_docs_pad, fasttext_model, input_dim,
                            embedding_dims)

# Word2vec Neural Network


# preprocessing
def text_preprocess(sentence):
    #stopwords = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    text_list = tokenizer.tokenize(sentence.lower())
    #text_list = [w for w in text_list if w not in stopwords]
    text_list = [w for w in text_list if w.isalpha()]
    txt = " ".join([w for w in text_list])

    return txt


min_count = 2
max_len = 70

pre_text = [text_preprocess(x) for x in data_full.text]
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(pre_text)

num_words = sum(
    [1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=True, filters='')
tokenizer.fit_on_texts(pre_text)
f_w2v_text_seq = tokenizer.texts_to_sequences(pre_text)
f_w2v_text_pad = pad_sequences(sequences=f_w2v_text_seq, maxlen=max_len)

word_index = tokenizer.word_index


def get_embedding_matrix(word_index, word2vec):
    # create an embedding matrix for the words we have in the dataset
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in tqdm(word_index.items()):
        try:
            embedding_vector = word2vec[word]
        except:
            continue
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


w2v_embedding_matrix = get_embedding_matrix(word_index, word2vec)
glv_embedding_matrix = get_embedding_matrix(word_index, glove2vec)

print(data_full.text[0])
print(pre_text[0])
print(f_w2v_text_pad[0])
print(len(word_index))
print(word_index['process'])
print(word2vec['process'])
print(w2v_embedding_matrix[3351])
print(glove2vec['process'])
print(glv_embedding_matrix[3351])


def lstm_model(input_dim, embedding_dims, max_len, embedding_matrix):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=input_dim,
            output_dim=embedding_dims,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.8))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


# w2v lstm model
input_dim = len(word_index) + 1
embedding_dims = 300
max_len = 70
embedding_matrix = w2v_embedding_matrix
p_w2v_lstm = apply_nn_model(f_w2v_text_pad, lstm_model, input_dim,
                            embedding_dims, max_len, embedding_matrix)

p_w2v_lstm.to_pickle('../model/p_w2v_lstm.pkl')
#p_w2v_lstm = pd.read_pickle('../model/p_w2v_lstm.pkl')

input_dim = len(word_index) + 1
embedding_dims = 300
max_len = 70
embedding_matrix = glv_embedding_matrix
p_glv_lstm = apply_nn_model(f_w2v_text_pad, lstm_model, input_dim,
                            embedding_dims, max_len, embedding_matrix)

p_glv_lstm.to_pickle('../model/p_glv_lstm.pkl')

#p_glv_lstm = pd.read_pickle('../model/p_glv_lstm.pkl')


def bi_lstm_model(input_dim, embedding_dims, max_len, embedding_matrix):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=input_dim,
            output_dim=embedding_dims,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.8))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


# bi lstm model
input_dim = len(word_index) + 1
embedding_dims = 300
max_len = 70
embedding_matrix = w2v_embedding_matrix
p_w2v_bi_lstm = apply_nn_model(f_w2v_text_pad, bi_lstm_model, input_dim,
                               embedding_dims, max_len, embedding_matrix)

p_w2v_bi_lstm.to_pickle('../model/p_w2v_bi_lstm.pkl')
#p_w2v_bi_lstm = pd.read_pickle('../model/p_w2v_bi_lstm.pkl')

input_dim = len(word_index) + 1
embedding_dims = 300
max_len = 70
embedding_matrix = glv_embedding_matrix
p_glv_bi_lstm = apply_nn_model(f_w2v_text_pad, bi_lstm_model, input_dim,
                               embedding_dims, max_len, embedding_matrix)

p_glv_bi_lstm.to_pickle('../model/p_glv_bi_lstm.pkl')

#p_glv_bi_lstm = pd.read_pickle('../model/p_glv_bi_lstm.pkl')


def gru_model(input_dim, embedding_dims, max_len, embedding_matrix):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=input_dim,
            output_dim=embedding_dims,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.8))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


# gru model

input_dim = len(word_index) + 1
embedding_dims = 300
max_len = 70
embedding_matrix = w2v_embedding_matrix
p_w2v_gru = apply_nn_model(f_w2v_text_pad, gru_model, input_dim,
                           embedding_dims, max_len, embedding_matrix)

p_w2v_gru.to_pickle('../model/p_w2v_gru.pkl')
#p_w2v_gru = pd.read_pickle('../model/p_w2v_gru.pkl')

input_dim = len(word_index) + 1
embedding_dims = 300
max_len = 70
embedding_matrix = glv_embedding_matrix
p_glv_gru = apply_nn_model(f_w2v_text_pad, gru_model, input_dim,
                           embedding_dims, max_len, embedding_matrix)

p_glv_gru.to_pickle('../model/p_glv_gru.pkl')
#p_glv_gru = pd.read_pickle('../model/p_glv_gru.pkl')

# # Ensembling & Stacking models

# ## Helper functions

import scipy.sparse

cv = StratifiedKFold(n_splits=5)


def apply_model(model, features, evaluate=True, predict=False):
    if any(map(lambda z: type(z) in [scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, scipy.sparse.coo_matrix], features)):
        hstack = scipy.sparse.hstack
    else:
        hstack = np.hstack

    f_all = hstack(features)
    f_train = f_all[np.nonzero(i_train)]
    f_test = f_all[np.nonzero(i_test)]

    p_cv = model_selection.cross_val_predict(
        model, f_train, y_train, cv=cv, method="predict_proba")
    q_cv = metrics.log_loss(y_train, p_cv)

    model.fit(f_train, y_train)

    p_train = model.predict_proba(f_train)
    q_train = metrics.log_loss(y_train, p_train)

    if evaluate:
        print(f"train log loss = {q_train:.5f}")
        print(f"   cv log loss = {q_cv:.5f}")

    if predict:
        p_test = model.predict_proba(f_test)
        p_full = np.concatenate((p_cv, p_test), axis=0)
        return pd.DataFrame(p_full)


# Confusion Matrix
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## First level Predictions

lr_param_grid = {'C': [0.1, 0.3, 1.0, 3.0, 10.0], 'penalty': ['l1', 'l2']}

nb_param_grid = {'alpha': [0.01, 0.03, 0.1, 0.3, 1, 10, 100]}

rf_param_grid = {
    'n_estimators': [120, 300, 500, 800],
    'max_features': ['log2', 'sqrt', None],
    'max_depth': [5, 8, 15, 25, 30, None],
    'min_samples_split': [0.1, 0.3, 1.0, 2, 5],
    'min_samples_leaf': [1, 2, 5, 10, 20]
}


def get_best_parameters(clf, param_grid, features, y_train):
    if any(map(lambda z: type(z) in [scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, scipy.sparse.coo_matrix], features)):
        hstack = scipy.sparse.hstack
    else:
        hstack = np.hstack

    f_all = hstack(features)
    f_train = f_all[np.nonzero(i_train)]

    model = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='neg_log_loss',
        cv=cv,
        verbose=0,
        n_jobs=-1)

    model.fit(f_train, y_train)
    best_parameters = model.best_estimator_.get_params()

    return best_parameters


# ### models using Basic features

# lr model
#lr_params = get_best_parameters(LogisticRegression(), lr_param_grid, f_basic, y_train)
#for key, value in lr_params.items():
#    print(key, value)
p_lr_basic = apply_model(LogisticRegression(), f_basic, predict=True)

# nb model
p_nb_basic = apply_model(MultinomialNB(), f_basic, predict=True)

# rf model
p_rf_basic = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    f_basic,
    predict=True)

# et model
p_et_basic = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    f_basic,
    predict=True)

# ### model using word based 6 features

# In[50]:

# lr model
p_lr_tfidf = apply_model(LogisticRegression(), (f_tfidf, ), predict=True)
p_lr_svd_tfidf = apply_model(
    LogisticRegression(), (f_svd_tfidf, ), predict=True)
p_lr_scl_svd_tfidf = apply_model(
    LogisticRegression(), (f_scl_svd_tfidf, ), predict=True)

p_lr_count = apply_model(LogisticRegression(), (f_count, ), predict=True)
p_lr_svd_count = apply_model(
    LogisticRegression(), (f_svd_count, ), predict=True)
p_lr_scl_svd_count = apply_model(
    LogisticRegression(), (f_scl_svd_count, ), predict=True)

print('---------------------------------')

# nb model
p_nb_tfidf = apply_model(MultinomialNB(), (f_tfidf, ), predict=True)
#p_nb_svd_tfidf = apply_model(MultinomialNB(), (f_svd_tfidf, ), predict=True)
#p_nb_scl_svd_tfidf = apply_model(MultinomialNB(), (f_scl_svd_tfidf, ), predict=True)

p_nb_count = apply_model(MultinomialNB(), (f_count, ), predict=True)
#p_nb_svd_count = apply_model(MultinomialNB(), (f_svd_count, ), predict=True)
#p_nb_scl_svd_count = apply_model(MultinomialNB(), (f_scl_svd_count, ), predict=True)

print('---------------------------------')

# rf model
p_rf_tfidf = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_tfidf, ),
    predict=True)
p_rf_svd_tfidf = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_svd_tfidf, ),
    predict=True)
p_rf_scl_svd_tfidf = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_tfidf, ),
    predict=True)

p_rf_count = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_count, ),
    predict=True)
p_rf_svd_count = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_svd_count, ),
    predict=True)
p_rf_scl_svd_count = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_count, ),
    predict=True)

print('---------------------------------')

# et model
p_et_tfidf = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_tfidf, ),
    predict=True)
p_et_svd_tfidf = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_svd_tfidf, ),
    predict=True)
p_et_scl_svd_tfidf = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_tfidf, ),
    predict=True)

p_et_count = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_count, ),
    predict=True)
p_et_svd_count = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_svd_count, ),
    predict=True)
p_et_scl_svd_count = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_count, ),
    predict=True)

# ### model using word stems based 6 features

# In[51]:

# lr model
p_lr_tfidf_s = apply_model(LogisticRegression(), (f_tfidf_s, ), predict=True)
p_lr_svd_tfidf_s = apply_model(
    LogisticRegression(), (f_svd_tfidf_s, ), predict=True)
p_lr_scl_svd_tfidf_s = apply_model(
    LogisticRegression(), (f_scl_svd_tfidf_s, ), predict=True)

p_lr_count_s = apply_model(LogisticRegression(), (f_count_s, ), predict=True)
p_lr_svd_count_s = apply_model(
    LogisticRegression(), (f_svd_count_s, ), predict=True)
p_lr_scl_svd_count_s = apply_model(
    LogisticRegression(), (f_scl_svd_count_s, ), predict=True)

print('---------------------------------')

# nb model
p_nb_tfidf_s = apply_model(MultinomialNB(), (f_tfidf_s, ), predict=True)
#p_nb_svd_tfidf_s = apply_model(MultinomialNB(), (f_svd_tfidf_s, ), predict=True)
#p_nb_scl_svd_tfidf_s = apply_model(MultinomialNB(), (f_scl_svd_tfidf_s, ), predict=True)

p_nb_count_s = apply_model(MultinomialNB(), (f_count_s, ), predict=True)
#p_nb_svd_count_s = apply_model(MultinomialNB(), (f_svd_count_s, ), predict=True)
#p_nb_scl_svd_count_s = apply_model(MultinomialNB(), (f_scl_svd_count_s, ), predict=True)

print('---------------------------------')

# rf model
p_rf_tfidf_s = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_tfidf_s, ),
    predict=True)
p_rf_svd_tfidf_s = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_svd_tfidf_s, ),
    predict=True)
p_rf_scl_svd_tfidf_s = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_tfidf_s, ),
    predict=True)

p_rf_count_s = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_count_s, ),
    predict=True)
p_rf_svd_count_s = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_svd_count_s, ),
    predict=True)
p_rf_scl_svd_count_s = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_count_s, ),
    predict=True)

print('---------------------------------')

# et model
p_et_tfidf_s = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_tfidf_s, ),
    predict=True)
p_et_svd_tfidf_s = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_svd_tfidf_s, ),
    predict=True)
p_et_scl_svd_tfidf_s = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_tfidf_s, ),
    predict=True)

p_et_count_s = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_count_s, ),
    predict=True)
p_et_svd_count_s = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_svd_count_s, ),
    predict=True)
p_et_scl_svd_count_s = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_count_s, ),
    predict=True)

# ### model using word lemmas based 6 features

# In[52]:

# lr model
p_lr_tfidf_l = apply_model(LogisticRegression(), (f_tfidf_l, ), predict=True)
p_lr_svd_tfidf_l = apply_model(
    LogisticRegression(), (f_svd_tfidf_l, ), predict=True)
p_lr_scl_svd_tfidf_l = apply_model(
    LogisticRegression(), (f_scl_svd_tfidf_l, ), predict=True)

p_lr_count_l = apply_model(LogisticRegression(), (f_count_l, ), predict=True)
p_lr_svd_count_l = apply_model(
    LogisticRegression(), (f_svd_count_l, ), predict=True)
p_lr_scl_svd_count_l = apply_model(
    LogisticRegression(), (f_scl_svd_count_l, ), predict=True)

print('---------------------------------')

# nb model
p_nb_tfidf_l = apply_model(MultinomialNB(), (f_tfidf_l, ), predict=True)
#p_nb_svd_tfidf_l = apply_model(MultinomialNB(), (f_svd_tfidf_l, ), predict=True)
#p_nb_scl_svd_tfidf_l = apply_model(MultinomialNB(), (f_scl_svd_tfidf_l, ), predict=True)

p_nb_count_l = apply_model(MultinomialNB(), (f_count_l, ), predict=True)
#p_nb_svd_count_l = apply_model(MultinomialNB(), (f_svd_count_l, ), predict=True)
#p_nb_scl_svd_count_l = apply_model(MultinomialNB(), (f_scl_svd_count_l, ), predict=True)

print('---------------------------------')

# rf model
p_rf_tfidf_l = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_tfidf_l, ),
    predict=True)
p_rf_svd_tfidf_l = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_svd_tfidf_l, ),
    predict=True)
p_rf_scl_svd_tfidf_l = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_tfidf_l, ),
    predict=True)

p_rf_count_l = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_count_l, ),
    predict=True)
p_rf_svd_count_l = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_svd_count_l, ),
    predict=True)
p_rf_scl_svd_count_l = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_count_l, ),
    predict=True)

print('---------------------------------')

# et model
p_et_tfidf_l = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_tfidf_l, ),
    predict=True)
p_et_svd_tfidf_l = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_svd_tfidf_l, ),
    predict=True)
p_et_scl_svd_tfidf_l = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_tfidf_l, ),
    predict=True)

p_et_count_l = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_count_l, ),
    predict=True)
p_et_svd_count_l = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_svd_count_l, ),
    predict=True)
p_et_scl_svd_count_l = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_count_l, ),
    predict=True)

# ### model using char based 6 features

# In[53]:

# lr model
p_lr_tfidf_c = apply_model(LogisticRegression(), (f_tfidf_c, ), predict=True)
p_lr_svd_tfidf_c = apply_model(
    LogisticRegression(), (f_svd_tfidf_c, ), predict=True)
p_lr_scl_svd_tfidf_c = apply_model(
    LogisticRegression(), (f_scl_svd_tfidf_c, ), predict=True)

p_lr_count_c = apply_model(LogisticRegression(), (f_count_c, ), predict=True)
p_lr_svd_count_c = apply_model(
    LogisticRegression(), (f_svd_count_c, ), predict=True)
p_lr_scl_svd_count_c = apply_model(
    LogisticRegression(), (f_scl_svd_count_c, ), predict=True)

print('---------------------------------')

# nb model
p_nb_tfidf_c = apply_model(MultinomialNB(), (f_tfidf_c, ), predict=True)
#p_nb_svd_tfidf_c = apply_model(MultinomialNB(), (f_svd_tfidf_c, ), predict=True)
#p_nb_scl_svd_tfidf_c = apply_model(MultinomialNB(), (f_scl_svd_tfidf_c, ), predict=True)

p_nb_count_c = apply_model(MultinomialNB(), (f_count_c, ), predict=True)
#p_nb_svd_count_c = apply_model(MultinomialNB(), (f_svd_count_c, ), predict=True)
#p_nb_scl_svd_count_c = apply_model(MultinomialNB(), (f_scl_svd_count_c, ), predict=True)

print('---------------------------------')

# rf model
p_rf_tfidf_c = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_tfidf_c, ),
    predict=True)
p_rf_svd_tfidf_c = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_svd_tfidf_c, ),
    predict=True)
p_rf_scl_svd_tfidf_c = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_tfidf_c, ),
    predict=True)

p_rf_count_c = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_count_c, ),
    predict=True)
p_rf_svd_count_c = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_svd_count_c, ),
    predict=True)
p_rf_scl_svd_count_c = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_count_c, ),
    predict=True)

print('---------------------------------')

# et model
p_et_tfidf_c = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_tfidf_c, ),
    predict=True)
p_et_svd_tfidf_c = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_svd_tfidf_c, ),
    predict=True)
p_et_scl_svd_tfidf_c = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_tfidf_c, ),
    predict=True)

p_et_count_c = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_count_c, ),
    predict=True)
p_et_svd_count_c = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_svd_count_c, ),
    predict=True)
p_et_scl_svd_count_c = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_svd_count_c, ),
    predict=True)

# ### models using Markov event based features

# In[54]:

# lr model
p_lr_markov = apply_model(
    LogisticRegression(), (f_markov_orders, ), predict=True)

# nb model
#p_nb_markov = apply_model(MultinomialNB(), (f_markov_orders, ), predict=True)

# rf model
p_rf_markov = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_markov_orders, ),
    predict=True)

# et model
p_et_markov = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_markov_orders, ),
    predict=True)

# ### models using Sentence vector features

# In[55]:

# lr model
p_lr_sent2v = apply_model(LogisticRegression(), (f_sent2v, ), predict=True)

# nb model
#p_nb_sent2v = apply_model(MultinomialNB(), (f_sent2v, ), predict=True)

# rf model
p_rf_sent2v = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_sent2v, ), predict=True)

# et model
p_et_sent2v = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_sent2v, ), predict=True)

# ## Second-Level Predictions

# non-negative
f_all_nne_features = f_basic + (
    p_lr_basic, p_nb_basic, p_rf_basic, p_et_basic, p_lr_tfidf, p_lr_svd_tfidf,
    p_lr_scl_svd_tfidf, p_lr_count, p_lr_svd_count, p_lr_scl_svd_count,
    p_nb_tfidf, p_nb_count, p_rf_tfidf, p_rf_svd_tfidf, p_rf_scl_svd_tfidf,
    p_rf_count, p_rf_svd_count, p_rf_scl_svd_count, p_et_tfidf, p_et_svd_tfidf,
    p_et_scl_svd_tfidf, p_et_count, p_et_svd_count, p_et_scl_svd_count,
    p_lr_tfidf_s, p_lr_svd_tfidf_s, p_lr_scl_svd_tfidf_s, p_lr_count_s,
    p_lr_svd_count_s, p_lr_scl_svd_count_s, p_nb_tfidf_s, p_nb_count_s,
    p_rf_tfidf_s, p_rf_svd_tfidf_s, p_rf_scl_svd_tfidf_s, p_rf_count_s,
    p_rf_svd_count_s, p_rf_scl_svd_count_s, p_et_tfidf_s, p_et_svd_tfidf_s,
    p_et_scl_svd_tfidf_s, p_et_count_s, p_et_svd_count_s, p_et_scl_svd_count_s,
    p_lr_tfidf_l, p_lr_svd_tfidf_l, p_lr_scl_svd_tfidf_l, p_lr_count_l,
    p_lr_svd_count_l, p_lr_scl_svd_count_l, p_nb_tfidf_l, p_nb_count_l,
    p_rf_tfidf_l, p_rf_svd_tfidf_l, p_rf_scl_svd_tfidf_l, p_rf_count_l,
    p_rf_svd_count_l, p_rf_scl_svd_count_l, p_et_tfidf_l, p_et_svd_tfidf_l,
    p_et_scl_svd_tfidf_l, p_et_count_l, p_et_svd_count_l, p_et_scl_svd_count_l,
    p_lr_tfidf_c, p_lr_svd_tfidf_c, p_lr_scl_svd_tfidf_c, p_lr_count_c,
    p_lr_svd_count_c, p_lr_scl_svd_count_c, p_nb_tfidf_c, p_nb_count_c,
    p_rf_tfidf_c, p_rf_svd_tfidf_c, p_rf_scl_svd_tfidf_c, p_rf_count_c,
    p_rf_svd_count_c, p_rf_scl_svd_count_c, p_et_tfidf_c, p_et_svd_tfidf_c,
    p_et_scl_svd_tfidf_c, p_et_count_c, p_et_svd_count_c, p_et_scl_svd_count_c,
    p_lr_markov, p_rf_markov, p_et_markov, p_lr_sent2v, p_lr_sent2v,
    p_lr_sent2v, p_nn, p_sent2v_nn, p_fasttext, p_w2v_lstm, p_w2v_bi_lstm,
    p_w2v_gru, p_glv_lstm, p_glv_bi_lstm, p_glv_gru)

f_all_features = f_all_nne_features + (
    f_svd_tfidf, f_scl_svd_tfidf, f_svd_count, f_scl_svd_count, f_svd_tfidf_s,
    f_scl_svd_tfidf_s, f_svd_count_s, f_scl_svd_count_s, f_svd_tfidf_l,
    f_scl_svd_tfidf_l, f_svd_count_l, f_scl_svd_count_l, f_svd_tfidf_c,
    f_scl_svd_tfidf_c, f_svd_count_c, f_scl_svd_count_c, f_markov_orders,
    f_sent2v)

p_xgb_all = apply_model(
    xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.5,
        missing=-999,
        nthread=-1,
        silent=1,
        objective='multi:softprob',
        seed=2017),
    f_all_features,
    predict=True)

#p_lr_all = apply_model(
#    CalibratedClassifierCV(LogisticRegression(max_iter=10), method="isotonic"),
#    f_all_features,
#    predict=True)
#p_nb_all = apply_model(
#    CalibratedClassifierCV(MultinomialNB(), method="isotonic"),
#    f_all_nne_features,
#    predict=True)
#p_rf_all = apply_model(
#    CalibratedClassifierCV(
#        RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
#        method="isotonic"),
#    f_all_features,
#    predict=True)
#p_et_all = apply_model(
#    CalibratedClassifierCV(
#        ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
#        method="isotonic"),
#    f_all_features,
#    predict=True)

#print('p_xgb_all_all')
#
#p_xgb_all_all = apply_model(
#    xgb.XGBClassifier(
#        learning_rate=0.1,
#        n_estimators=100,
#        max_depth=5,
#        min_child_weight=1,
#        subsample=0.8,
#        colsample_bytree=0.5,
#        missing=-999,
#        nthread=-1,
#        silent=1,
#        objective='multi:softprob',
#        seed=2017),
#    f_all_features + (p_xgb_all, p_lr_all, p_nb_all, p_rf_all, p_et_all),
#    predict=True)

import os.path


def make_submission(predictions, submission_path, file_name):
    file_name = os.path.join(submission_path, os.path.basename(file_name))
    predictions = predictions.copy()
    predictions.columns = ['EAP', 'HPL', 'MWS']
    pd.concat(
        (data_test["id"].reset_index(drop=True),
         predictions[i_test].reset_index(drop=True)),
        axis=1).to_csv(
            file_name, index=False)


make_submission(p_xgb_all, '../result', "p_xgb_all-20171211-02.csv")
