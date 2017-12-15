#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import string
from tqdm import tqdm
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from imageio import imread
from wordcloud import WordCloud, STOPWORDS

from afinn import Afinn

import gensim
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics, model_selection
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.pipeline import make_pipeline
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

punctuations = [{
    "id": 1,
    "p": "[;:]"
}, {
    "id": 2,
    "p": "[,.]"
}, {
    "id": 3,
    "p": "[?]"
}, {
    "id": 4,
    "p": "[\']"
}, {
    "id": 5,
    "p": "[\"]"
}, {
    "id": 6,
    "p": "[;:,.?\'\"]"
}]


def get_words(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    word_list = tokenizer.tokenize(text)
    return word_list


def num_words(text):
    return len(get_words(text))


def num_chars(text):
    return len(text)


def mean_len_words(text):
    return np.mean([len(w) for w in get_words(text)])


def first_word_len(text):
    return len(get_words(text)[0])


def last_word_len(text):
    return len(get_words(text)[-1])


def num_unique_words(text):
    return len(list(set(get_words(text))))


def num_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    return len([w for w in get_words(text) if w in stopwords])


def num_punctuations(text):
    word_list = nltk.word_tokenize(text)
    return len([w for w in word_list if w in string.punctuation])


def num_sets_punctuations(text):
    word_list = nltk.word_tokenize(text)

    punc_nums = np.zeros(len(punctuations))
    for i, punc in enumerate(punctuations):
        #punc_nums[i] = len([w for w in word_list if bool(re.search(punc["p"], w))]) * 100.0 / len(word_list)
        punc_nums[i] = len(
            [w for w in word_list if bool(re.search(punc["p"], w))])
    return punc_nums


def num_upper_words(text):
    return len([w for w in get_words(text) if w.isupper()])


def num_title_words(text):
    return len([w for w in get_words(text) if w.istitle()])


def num_unknown_symbols(text):
    symbols_known = string.ascii_letters + string.digits + string.punctuation
    return sum([not x in symbols_known for x in text])


def num_noun_words(text):
    pos_list = nltk.pos_tag(get_words(text))
    return len([w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])


def num_adj_words(text):
    pos_list = nltk.pos_tag(get_words(text))
    return len([w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')])


def num_verbs_words(text):
    pos_list = nltk.pos_tag(get_words(text))
    return len([
        w for w in pos_list
        if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    ])


def symbol_id(x):
    symbols = [
        x for x in string.ascii_letters + string.digits + string.punctuation
    ]
    return np.where(np.array(symbols) == x)[0][0]


f_num_words = np.array([num_words(text) for text in data_full.text]).reshape(
    data_full.shape[0], -1)
f_num_chars = np.array([num_chars(text) for text in data_full.text]).reshape(
    data_full.shape[0], -1)
f_mean_len_words = np.array([mean_len_words(text)
                             for text in data_full.text]).reshape(
                                 data_full.shape[0], -1)
f_first_word_len = np.array([first_word_len(text)
                             for text in data_full.text]).reshape(
                                 data_full.shape[0], -1)
f_last_word_len = np.array([last_word_len(text)
                            for text in data_full.text]).reshape(
                                data_full.shape[0], -1)
f_first_symbol_id = np.array([symbol_id(text[0])
                              for text in data_full.text]).reshape(
                                  data_full.shape[0], -1)
f_last_symbol_id = np.array([symbol_id(text[-1])
                             for text in data_full.text]).reshape(
                                 data_full.shape[0], -1)

f_num_unique_words = np.array(
    [num_unique_words(text)
     for text in data_full.text]).reshape(data_full.shape[0], -1)
f_num_stopwords = np.array([num_stopwords(text)
                            for text in data_full.text]).reshape(
                                data_full.shape[0], -1)
f_num_punctuations = np.array(
    [num_punctuations(text)
     for text in data_full.text]).reshape(data_full.shape[0], -1)
f_num_sets_punctuations = np.array(
    [num_sets_punctuations(text)
     for text in data_full.text]).reshape(data_full.shape[0], -1)
f_num_upper_words = np.array(
    [num_upper_words(text)
     for text in data_full.text]).reshape(data_full.shape[0], -1)
f_num_title_words = np.array(
    [num_title_words(text)
     for text in data_full.text]).reshape(data_full.shape[0], -1)
f_num_unknown_symbols = np.array(
    [num_unknown_symbols(text)
     for text in data_full.text]).reshape(data_full.shape[0], -1)

f_num_noun_words = np.array([num_noun_words(text)
                             for text in data_full.text]).reshape(
                                 data_full.shape[0], -1)
f_num_adj_words = np.array([num_adj_words(text)
                            for text in data_full.text]).reshape(
                                data_full.shape[0], -1)
f_num_verbs_words = np.array(
    [num_verbs_words(text)
     for text in data_full.text]).reshape(data_full.shape[0], -1)

f_fra_unique_words = f_num_unique_words / f_num_words
f_fra_stopwords = f_num_stopwords / f_num_words
f_fra_punctuations = f_num_punctuations / f_num_words
f_fra_sets_punctuations = f_num_sets_punctuations / f_num_words
f_fra_upper_words = f_num_upper_words / f_num_words
f_fra_title_words = f_num_title_words / f_num_words
f_fra_unknown_symbols = f_num_unknown_symbols / f_num_chars
f_fra_noun_words = f_num_noun_words / f_num_words
f_fra_adj_words = f_num_adj_words / f_num_words
f_fra_verbs_words = f_num_verbs_words / f_num_words

f_basic = (f_num_words, f_num_chars, f_mean_len_words, f_first_word_len,
           f_last_word_len, f_first_symbol_id, f_last_symbol_id,
           f_num_unique_words, f_num_stopwords, f_num_punctuations,
           f_num_sets_punctuations, f_num_upper_words, f_num_title_words,
           f_num_unknown_symbols, f_num_noun_words, f_num_adj_words,
           f_num_verbs_words, f_fra_unique_words, f_fra_stopwords,
           f_fra_punctuations, f_fra_sets_punctuations, f_fra_upper_words,
           f_fra_title_words, f_fra_unknown_symbols, f_fra_noun_words,
           f_fra_adj_words, f_fra_verbs_words)

## before save
#p_lr_basic = apply_model(LogisticRegression(), f_basic, predict=True)
#
## after save
#np.save('../model/f_basic', np.hstack(f_basic))

f_basic = np.load('../model/f_basic.npy')

#
#p_lr_basic = apply_model(LogisticRegression(), (f_basic, ), predict=True)


# bigram clouds
class WordCloudIntersection():
    def __init__(self,
                 stopwords=list(),
                 punctuation=list(),
                 stemmer=None,
                 ngram=1):
        self.stopwords = stopwords
        self.punctuation = punctuation
        self.remove = self.stopwords + self.punctuation
        self.clouds = dict()
        self.texts = dict()
        self.stemmer = stemmer
        self.ngram = ngram

    def find_ngrams(self, input_list, n):
        return [
            " ".join(list(i))
            for i in zip(* [input_list[i:] for i in range(n)])
        ]

    # It would be much  more correct to call this function 'get_tokens'
    # it extracts not only words, but n-grams as well
    def get_words(self, text):
        words = nltk.tokenize.word_tokenize(text)
        words = [w for w in words if not w in self.remove]
        if not self.stemmer is None:
            words = [self.stemmer.stem(w) for w in words]

        if self.ngram > 1:
            words = self.find_ngrams(words, self.ngram)
        return words

    # Jaccard distance again
    def relative_intersection(self, x, y):
        try:
            return len(x & y) / len(x | y)
        except:
            return 0.0

    def fit(self, x, categories, data_train, data_test=None):
        cat_names = np.unique(data_train[categories])

        text_train = " ".join(list(data_train[x]))
        text_test = ""
        if not data_test is None:
            text_test = " ".join(list(data_test[x]))

        # Tokens presenting in both train and test data
        words_unique = self.get_words((text_train + text_test).lower())

        for cat in cat_names:
            self.texts[cat] = (
                " ".join(list(data_train[x][data_train[categories] == cat]))
            ).lower()
            words = self.get_words(self.texts[cat])
            self.clouds[cat] = pd.value_counts(words)

        # use only tokens presented in both train and test data,
        # feature will force your model to overfit to the train data otherwise
        for cat in cat_names:
            self.clouds[cat] = self.clouds[cat][list(
                set(self.clouds[cat].index) & set(words_unique))]

        # Keep only author-specific tokens
        for cat in cat_names:
            key_leftover = list(set(cat_names) - set([cat]))
            bigrams_other = set(self.clouds[key_leftover[0]].index) | set(
                self.clouds[key_leftover[1]].index)
            self.clouds[cat] = self.clouds[cat][list(
                set(self.clouds[cat].index) - bigrams_other)]

    def transform(self, x, data):
        intersection = dict()
        prefix = '_intersect_'
        if self.ngram > 1:
            prefix = '%s-gram%s' % (self.ngram, prefix)
        else:
            prefix = 'word' + prefix
        for key in self.clouds.keys():
            category_words_set = set(self.clouds[key].index)
            intersection[prefix + key] = list()
            for text in data[x]:
                unique_words = set(self.get_words(text.lower()))
                fraction = self.relative_intersection(unique_words,
                                                      category_words_set)
                intersection[prefix + key].append(fraction)
        return pd.DataFrame(intersection)


"""
stopwords = nltk.corpus.stopwords.words('english')
t_bigci = WordCloudIntersection(
    stopwords=stopwords,
    punctuation=list(string.punctuation),
    stemmer=nltk.stem.SnowballStemmer('english'),
    ngram=2)
t_bigci.fit(
    x='text', categories='author', data_train=data_train, data_test=data_test)

f_train_big_intersections = t_bigci.transform(x='text', data=data_train)
f_test_big_intersections = t_bigci.transform(x='text', data=data_test)
f_big_intersections = pd.DataFrame(
    np.concatenate(
        (f_train_big_intersections, f_test_big_intersections), axis=0))
print(f_big_intersections.head())
f_big_intersections.to_pickle('../model/f_big_intersections.pkl')
"""
f_big_intersections = pd.read_pickle('../model/f_big_intersections.pkl')
"""
t_trigci = WordCloudIntersection(
    stopwords=stopwords,
    punctuation=list(string.punctuation),
    stemmer=nltk.stem.SnowballStemmer('english'),
    ngram=3)
t_trigci.fit(
    x='text', categories='author', data_train=data_train, data_test=data_test)

f_train_trig_intersections = t_trigci.transform(x='text', data=data_train)
f_test_trig_intersections = t_trigci.transform(x='text', data=data_test)
f_trig_intersections = pd.DataFrame(
    np.concatenate(
        (f_train_trig_intersections, f_test_trig_intersections), axis=0))
print(f_trig_intersections.head())
f_trig_intersections.to_pickle('../model/f_trig_intersections.pkl')
"""
f_trig_intersections = pd.read_pickle('../model/f_trig_intersections.pkl')

from sklearn.preprocessing import normalize

f_nor_big_intersections = normalize(f_big_intersections)
#f_1000_big_intersections = f_big_intersections * 1000
#f_10000_big_intersections = f_big_intersections * 10000

f_nor_trig_intersections = normalize(f_trig_intersections)
#f_1000_trig_intersections = f_trig_intersections * 1000
#f_10000_trig_intersections = f_trig_intersections * 10000

n_components = 40

# TFIDF features
t_tfidf = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 3),
    min_df=3)  # token_pattern=r'\w{1,}'
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
t_count = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=3)
f_count = t_count.fit_transform(data_full.text)

# Counters with SVD features
t_svd_count = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_count = t_svd_count.fit_transform(f_count.astype(float))  # float

# Counter with SVD and Scale features
t_scl_svd_count = StandardScaler()
f_scl_svd_count = t_scl_svd_count.fit_transform(f_svd_count)

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


# TFIDF features
t_tfidf_s = TfidfVectorizer(
    tokenizer=tokenize_s, stop_words="english", ngram_range=(1, 3), min_df=3)
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
    tokenizer=tokenize_s, stop_words="english", ngram_range=(1, 3), min_df=3)
f_count_s = t_count_s.fit_transform(data_full.text)

# Counters with SVD features
t_svd_count_s = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_count_s = t_svd_count_s.fit_transform(f_count_s.astype(float))

# Counters with SVD and Scale features
t_scl_svd_count_s = StandardScaler()
f_scl_svd_count_s = t_scl_svd_count_s.fit_transform(f_svd_count_s)

lemmatizer = nltk.stem.WordNetLemmatizer()


def tokenize_l(text):
    lemms = []
    for i, j in nltk.pos_tag(nltk.word_tokenize(text.lower())):
        if j[0].lower() in ['a', 'n', 'v']:
            lemms.append(lemmatizer.lemmatize(i, j[0].lower()))
        else:
            lemms.append(lemmatizer.lemmatize(i))
    return lemms


# TFIDF features
t_tfidf_l = TfidfVectorizer(
    tokenizer=tokenize_l, stop_words="english", ngram_range=(1, 3), min_df=3)
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
    tokenizer=tokenize_l, stop_words="english", ngram_range=(1, 3), min_df=3)
f_count_l = t_count_l.fit_transform(data_full.text)

# Counters with SVD features
t_svd_count_l = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_count_l = t_svd_count_l.fit_transform(f_count_l.astype(float))

# Counter with SVD and Scale features
t_scl_svd_count_l = StandardScaler()
f_scl_svd_count_l = t_scl_svd_count_l.fit_transform(f_svd_count_l)

# TFIDF features
t_tfidf_c = TfidfVectorizer(
    analyzer="char", stop_words="english", ngram_range=(1, 5), min_df=3)
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
    analyzer="char", stop_words="english", ngram_range=(1, 5), min_df=3)
f_count_c = t_count_c.fit_transform(data_full.text)

# Counter with SVD features
t_svd_count_c = TruncatedSVD(
    n_components=n_components, algorithm="arpack", random_state=2017)
f_svd_count_c = t_svd_count_c.fit_transform(f_count_c.astype(float))

# Counter with SVD and Scale features
t_scl_svd_count_c = StandardScaler()
f_scl_svd_count_c = t_scl_svd_count_c.fit_transform(f_svd_count_c)


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

print(f_markov_orders[:10])

#word2vec = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin', binary=True)

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


#glove2vec = loadGloveEmbeddings(glove_file)


def sent2vec(sentence, word2vec):
    #stopwords = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    text_list = tokenizer.tokenize(sentence.lower())
    #text_list = [w for w in text_list if w not in stopwords]
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


#w2v_sent2v = [sent2vec(x, word2vec) for x in data_full.text]
#f_w2v_sent2v = np.array(w2v_sent2v)
#glv_sent2v = [sent2vec(x, glove2vec) for x in data_full.text]
#f_glv_sent2v = np.array(glv_sent2v)

#np.save('../model/f_w2v_sent2v', f_w2v_sent2v)
#np.save('../model/f_glv_sent2v', f_glv_sent2v)
f_w2v_sent2v = np.load('../model/f_w2v_sent2v.npy')
f_glv_sent2v = np.load('../model/f_glv_sent2v.npy')

t_scl_w2v_sent2v = StandardScaler()
f_scl_w2v_sent2v = t_scl_w2v_sent2v.fit_transform(f_w2v_sent2v)
t_scl_glv_sent2v = StandardScaler()
f_scl_glv_sent2v = t_scl_glv_sent2v.fit_transform(f_glv_sent2v)

# Sentiment
afinn = Afinn()
sia = SentimentIntensityAnalyzer()


def get_senti_score(text):
    return afinn.score(text)


def sentiment_nltk(text):
    res = sia.polarity_scores(text)
    res_senti = []
    for i in res:
        res_senti.append(res[i])
    return res_senti


f_senti = np.array([get_senti_score(text) for text in data_full.text]).reshape(
    data_full.shape[0], -1)

f_nltk_senti = np.array([sentiment_nltk(text)
                         for text in data_full.text]).reshape(
                             data_full.shape[0], -1)


def apply_nn_model(features,
                   model_func,
                   input_dim=None,
                   embedding_dims=None,
                   max_len=None,
                   embedding_matrix=None,
                   epochs=25,
                   patience=2):

    n_splits = 5
    kf = model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=2017)

    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=patience, verbose=0, mode='auto')

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

        model.fit(
            x_dev,
            y_dev,
            batch_size=512,
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


# Separate punctuation from words
# Remove lower frequency words ( <= 2)
# Cut a longer document which contains 256 words
def fasttext_preprocess(text):
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
        doc = fasttext_preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))

    return docs


"""
min_count = 2
ft_max_len = 90  # docs mean len: 79.2660255264

docs = create_docs(data_full)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)

ft_num_words = sum(
    [1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=ft_num_words, lower=True, filters='')
tokenizer.fit_on_texts(docs)
f_ft_docs_seq = tokenizer.texts_to_sequences(docs)
f_ft_docs_pad = pad_sequences(sequences=f_ft_docs_seq, maxlen=ft_max_len)

print(ft_num_words, len(tokenizer.word_index), np.max(f_ft_docs_pad))
"""


def text_preprocess(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    word_list = tokenizer.tokenize(text.lower())
    word_list = [w for w in word_list if w.isalpha()]
    ret_text = " ".join([w for w in word_list])

    return ret_text


"""
min_count = 2
max_len = 70

pre_text = [text_preprocess(x) for x in data_full.text]
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(pre_text)

num_words = sum(
    [1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=True, filters='')
tokenizer.fit_on_texts(pre_text)
f_text_seq = tokenizer.texts_to_sequences(pre_text)
f_text_pad = pad_sequences(sequences=f_text_seq, maxlen=max_len)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

print(num_words, len(word_index), np.max(f_text_pad))
"""


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


"""
w2v_embedding_matrix = get_embedding_matrix(word_index, word2vec)
glv_embedding_matrix = get_embedding_matrix(word_index, glove2vec)
"""


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


#p_nn = apply_nn_model(
#    f_text_pad,
#    nn_model,
#    input_dim=min(num_words, len(word_index)) + 1,
#    embedding_dims=32,
#    max_len=max_len,
#    epochs=5,
#    patience=1)

print('p_nn')
#p_nn.to_pickle('../model/p_nn.pkl')
p_nn = pd.read_pickle('../model/p_nn.pkl')


# no use input_dim, embedding_dims, max_len, embedding_matrix
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

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


#p_w2v_sent2v_nn = apply_nn_model(f_scl_w2v_sent2v, sent2vec_model, epochs=100, patience=2)
#p_glv_sent2v_nn = apply_nn_model(f_scl_glv_sent2v, sent2vec_model, epochs=100, patience=2)

#p_w2v_sent2v_nn.to_pickle('../model/p_w2v_sent2v_nn.pkl')
#p_glv_sent2v_nn.to_pickle('../model/p_glv_sent2v_nn.pkl')

print('p_w2v_sent2v_nn')
p_w2v_sent2v_nn = pd.read_pickle('../model/p_w2v_sent2v_nn.pkl')
print('p_glv_sent2v_nn')
p_glv_sent2v_nn = pd.read_pickle('../model/p_glv_sent2v_nn.pkl')


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


#p_fasttext = apply_nn_model(
#    f_ft_docs_pad,
#    fasttext_model,
#    input_dim=np.max(f_ft_docs_pad) + 1,
#    embedding_dims=20,
#    max_len=ft_max_len,
#    embedding_matrix=None,
#    epochs=100,
#    patience=2)
#p_fasttext.to_pickle('../model/p_fasttext.pkl')

print('p_fasttext above')
p_fasttext = pd.read_pickle('../model/p_fasttext.pkl')


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

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


#p_w2v_lstm = apply_nn_model(
#    f_text_pad,
#    lstm_model,
#    input_dim=len(word_index) + 1,
#    embedding_dims=300,
#    max_len=max_len,
#    embedding_matrix=w2v_embedding_matrix,
#    epochs=100,
#    patience=5)

#p_w2v_lstm.to_pickle('../model/p_w2v_lstm.pkl')

#p_glv_lstm = apply_nn_model(
#    f_text_pad,
#    lstm_model,
#    input_dim=len(word_index) + 1,
#    embedding_dims=300,
#    max_len=max_len,
#    embedding_matrix=glv_embedding_matrix,
#    epochs=100,
#    patience=3)

#p_glv_lstm.to_pickle('../model/p_glv_lstm.pkl')

print('p_w2v_lstm')
p_w2v_lstm = pd.read_pickle('../model/p_w2v_lstm.pkl')
print('p_glv_lstm')
p_glv_lstm = pd.read_pickle('../model/p_glv_lstm.pkl')


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

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


#p_w2v_bi_lstm = apply_nn_model(
#    f_text_pad,
#    bi_lstm_model,
#    input_dim=len(word_index) + 1,
#    embedding_dims=300,
#    max_len=max_len,
#    embedding_matrix=w2v_embedding_matrix,
#    epochs=100,
#    patience=5)

#p_w2v_bi_lstm.to_pickle('../model/p_w2v_bi_lstm.pkl')

#p_glv_bi_lstm = apply_nn_model(
#    f_text_pad,
#    bi_lstm_model,
#    input_dim=len(word_index) + 1,
#    embedding_dims=300,
#    max_len=max_len,
#    embedding_matrix=glv_embedding_matrix,
#    epochs=100,
#    patience=3)

#p_glv_bi_lstm.to_pickle('../model/p_glv_bi_lstm.pkl')

print('p_w2v_bi_lstm')
p_w2v_bi_lstm = pd.read_pickle('../model/p_w2v_bi_lstm.pkl')
print('p_glv_bi_lstm')
p_glv_bi_lstm = pd.read_pickle('../model/p_glv_bi_lstm.pkl')


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
    model.add(
        GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


#p_w2v_gru = apply_nn_model(
#    f_text_pad,
#    gru_model,
#    input_dim=len(word_index) + 1,
#    embedding_dims=300,
#    max_len=max_len,
#    embedding_matrix=w2v_embedding_matrix,
#    epochs=100,
#    patience=5)

#p_w2v_gru.to_pickle('../model/p_w2v_gru.pkl')

#p_glv_gru = apply_nn_model(
#    f_text_pad,
#    gru_model,
#    input_dim=len(word_index) + 1,
#    embedding_dims=300,
#    max_len=max_len,
#    embedding_matrix=glv_embedding_matrix,
#    epochs=100,
#    patience=3)

#p_glv_gru.to_pickle('../model/p_glv_gru.pkl')

print('p_w2v_gru')
p_w2v_gru = pd.read_pickle('../model/p_w2v_gru.pkl')
print('p_glv_gru')
p_glv_gru = pd.read_pickle('../model/p_glv_gru.pkl')


# no use max_len
def ex_fasttext_model(input_dim, embedding_dims, max_len, embedding_matrix):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=input_dim,
            output_dim=embedding_dims,
            weights=[embedding_matrix]))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


#p_w2v_ex_fasttext = apply_nn_model(
#    f_text_pad,
#    ex_fasttext_model,
#    input_dim=len(word_index) + 1,
#    embedding_dims=300,
#    embedding_matrix=w2v_embedding_matrix,
#    epochs=100,
#    patience=3)
#p_w2v_ex_fasttext.to_pickle('../model/p_w2v_ex_fasttext.pkl')

#p_glv_ex_fasttext = apply_nn_model(
#    f_text_pad,
#    ex_fasttext_model,
#    input_dim=len(word_index) + 1,
#    embedding_dims=300,
#    embedding_matrix=glv_embedding_matrix,
#    epochs=100,
#    patience=3)
#p_glv_ex_fasttext.to_pickle('../model/p_glv_ex_fasttext.pkl')

print('p_w2v_ex_fasttext')
p_w2v_ex_fasttext = pd.read_pickle('../model/p_w2v_ex_fasttext.pkl')
print('p_glv_ex_fasttext above')
p_glv_ex_fasttext = pd.read_pickle('../model/p_glv_ex_fasttext.pkl')

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


# lr model
#lr_params = get_best_parameters(LogisticRegression(), lr_param_grid, f_basic, y_train)
#for key, value in lr_params.items():
#    print(key, value)
p_lr_basic = apply_model(LogisticRegression(), (f_basic, ), predict=True)

# nb model
p_nb_basic = apply_model(MultinomialNB(), (f_basic, ), predict=True)

# rf model
p_rf_basic = apply_model(
    RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_basic, ),
    predict=True)

# et model
p_et_basic = apply_model(
    ExtraTreesClassifier(n_estimators=300, max_depth=6, n_jobs=-1),
    (f_basic, ),
    predict=True)

# xgb model
p_xgb_basic = apply_model(
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
        seed=2017), (f_basic, ),
    predict=True)

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

# xgb model
p_xgb_svd_tfidf = apply_model(
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
        seed=2017), (f_svd_tfidf, ),
    predict=True)
p_xgb_scl_svd_tfidf = apply_model(
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
        seed=2017), (f_scl_svd_tfidf, ),
    predict=True)

p_xgb_svd_count = apply_model(
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
        seed=2017), (f_svd_count, ),
    predict=True)
p_xgb_scl_svd_count = apply_model(
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
        seed=2017), (f_scl_svd_count, ),
    predict=True)

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

# xgb model
p_xgb_svd_tfidf_s = apply_model(
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
        seed=2017), (f_svd_tfidf_s, ),
    predict=True)
p_xgb_scl_svd_tfidf_s = apply_model(
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
        seed=2017), (f_scl_svd_tfidf_s, ),
    predict=True)

p_xgb_svd_count_s = apply_model(
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
        seed=2017), (f_svd_count_s, ),
    predict=True)
p_xgb_scl_svd_count_s = apply_model(
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
        seed=2017), (f_scl_svd_count_s, ),
    predict=True)

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

# xgb model
p_xgb_svd_tfidf_l = apply_model(
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
        seed=2017), (f_svd_tfidf_l, ),
    predict=True)
p_xgb_scl_svd_tfidf_l = apply_model(
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
        seed=2017), (f_scl_svd_tfidf_l, ),
    predict=True)

p_xgb_svd_count_l = apply_model(
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
        seed=2017), (f_svd_count_l, ),
    predict=True)
p_xgb_scl_svd_count_l = apply_model(
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
        seed=2017), (f_scl_svd_count_l, ),
    predict=True)

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

# xgb model
p_xgb_svd_tfidf_c = apply_model(
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
        seed=2017), (f_svd_tfidf_c, ),
    predict=True)
p_xgb_scl_svd_tfidf_c = apply_model(
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
        seed=2017), (f_scl_svd_tfidf_c, ),
    predict=True)

p_xgb_svd_count_c = apply_model(
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
        seed=2017), (f_svd_count_c, ),
    predict=True)
p_xgb_scl_svd_count_c = apply_model(
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
        seed=2017), (f_scl_svd_count_c, ),
    predict=True)

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

# xgb model
p_xgb_markov = apply_model(
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
        seed=2017), (f_markov_orders, ),
    predict=True)

# lr model
p_lr_w2v_sent2v = apply_model(
    LogisticRegression(), (f_w2v_sent2v, ), predict=True)
p_lr_scl_w2v_sent2v = apply_model(
    LogisticRegression(), (f_scl_w2v_sent2v, ), predict=True)

p_lr_glv_sent2v = apply_model(
    LogisticRegression(), (f_glv_sent2v, ), predict=True)
p_lr_scl_glv_sent2v = apply_model(
    LogisticRegression(), (f_scl_glv_sent2v, ), predict=True)

# nb model

# rf model
p_rf_w2v_sent2v = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_w2v_sent2v, ),
    predict=True)
p_rf_scl_w2v_sent2v = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_w2v_sent2v, ),
    predict=True)

p_rf_glv_sent2v = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_glv_sent2v, ),
    predict=True)
p_rf_scl_glv_sent2v = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_scl_glv_sent2v, ),
    predict=True)

# et model
p_et_w2v_sent2v = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_w2v_sent2v, ),
    predict=True)
p_et_scl_w2v_sent2v = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_w2v_sent2v, ),
    predict=True)

p_et_glv_sent2v = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_glv_sent2v, ),
    predict=True)
p_et_scl_glv_sent2v = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_scl_glv_sent2v, ),
    predict=True)

# xgb model
p_xgb_w2v_sent2v = apply_model(
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
        seed=2017), (f_w2v_sent2v, ),
    predict=True)
p_xgb_scl_w2v_sent2v = apply_model(
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
        seed=2017), (f_scl_w2v_sent2v, ),
    predict=True)

p_xgb_glv_sent2v = apply_model(
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
        seed=2017), (f_glv_sent2v, ),
    predict=True)
p_xgb_scl_glv_sent2v = apply_model(
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
        seed=2017), (f_scl_glv_sent2v, ),
    predict=True)

# lr model
p_lr_big_intersections = apply_model(
    LogisticRegression(), (f_big_intersections, ), predict=True)
#p_lr_nor_big_intersections = apply_model(LogisticRegression(), (f_nor_big_intersections, ), predict=True)
p_lr_trig_intersections = apply_model(
    LogisticRegression(), (f_big_intersections, ), predict=True)
#p_lr_nor_trig_intersections = apply_model(LogisticRegression(), (f_nor_trig_intersections, ), predict=True)

#nb model
p_nb_big_intersections = apply_model(
    MultinomialNB(), (f_big_intersections, ), predict=True)
p_nb_trig_intersections = apply_model(
    MultinomialNB(), (f_trig_intersections, ), predict=True)
#p_nb_nor_big_intersections = apply_model(MultinomialNB(), (f_nor_big_intersections, ), predict=True)
#p_nb_nor_trig_intersections = apply_model(MultinomialNB(), (f_nor_trig_intersections, ), predict=True)

# rf model
#p_rf_big_intersections = apply_model(RandomForestClassifier(max_depth=6, n_jobs=-1),
#                                         (f_big_intersections * 100, ), predict=True)
#p_rf_trig_intersections = apply_model(RandomForestClassifier(max_depth=6, n_jobs=-1),
#                                         (f_trig_intersections * 100, ), predict=True)
#p_rf_nor_big_intersections = apply_model(RandomForestClassifier(max_depth=6, n_jobs=-1),
#                                         (f_nor_big_intersections, ), predict=True)
#p_rf_nor_trig_intersections = apply_model(RandomForestClassifier(max_depth=6, n_jobs=-1),
#                                         (f_nor_trig_intersections, ), predict=True)

# et model
p_et_big_intersections = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_big_intersections, ),
    predict=True)
p_et_trig_intersections = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_trig_intersections, ),
    predict=True)
#p_et_nor_big_intersections = apply_model(ExtraTreesClassifier(max_depth=6, n_jobs=-1),
#                                         (f_nor_big_intersections, ), predict=True)
#p_et_nor_trig_intersections = apply_model(ExtraTreesClassifier(max_depth=6, n_jobs=-1),
#                                         (f_nor_trig_intersections, ), predict=True)

# lr model
p_lr_senti = apply_model(LogisticRegression(), (f_senti, ), predict=True)
p_lr_nltk_senti = apply_model(
    LogisticRegression(), (f_nltk_senti, ), predict=True)

# nb model
#p_nb_senti = apply_model(MultinomialNB(), (f_senti, ), predict=True)

# rf model
p_rf_senti = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_senti, ), predict=True)
p_rf_nltk_senti = apply_model(
    RandomForestClassifier(max_depth=6, n_jobs=-1), (f_nltk_senti, ),
    predict=True)

# et model
p_et_senti = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_senti, ), predict=True)
p_et_nltk_senti = apply_model(
    ExtraTreesClassifier(max_depth=6, n_jobs=-1), (f_nltk_senti, ),
    predict=True)

# xgb model
#p_xgb_senti = apply_model(
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
#        seed=2017), (f_senti, ),
#    predict=True)
#p_xgb_nltk_senti = apply_model(
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
#        seed=2017), (f_nltk_senti, ),
#predict=True)

# non-negative
#f_all_nne_features = tuple(f_basic) + (
f_all_nne_features = (
    f_basic,
    p_lr_basic,
    p_nb_basic,
    p_rf_basic,
    p_et_basic,
    p_xgb_basic,
    p_lr_tfidf,
    p_lr_svd_tfidf,
    p_lr_scl_svd_tfidf,
    p_lr_count,
    #p_lr_svd_count,
    #p_lr_scl_svd_count,
    p_nb_tfidf,
    p_nb_count,
    #p_rf_tfidf,
    p_rf_svd_tfidf,
    p_rf_scl_svd_tfidf,
    #p_rf_count,
    #p_rf_svd_count,
    #p_rf_scl_svd_count,
    #p_et_tfidf,
    #p_et_svd_tfidf,
    #p_et_scl_svd_tfidf,
    #p_et_count,
    #p_et_svd_count,
    #p_et_scl_svd_count,
    p_xgb_svd_tfidf,
    p_xgb_scl_svd_tfidf,
    p_xgb_svd_count,
    p_xgb_scl_svd_count,
    p_lr_tfidf_s,
    p_lr_svd_tfidf_s,
    p_lr_scl_svd_tfidf_s,
    p_lr_count_s,
    p_lr_svd_count_s,
    p_lr_scl_svd_count_s,
    p_nb_tfidf_s,
    p_nb_count_s,
    #p_rf_tfidf_s,
    p_rf_svd_tfidf_s,
    p_rf_scl_svd_tfidf_s,
    #p_rf_count_s,
    p_rf_svd_count_s,
    p_rf_scl_svd_count_s,
    #p_et_tfidf_s,
    #p_et_svd_tfidf_s,
    #p_et_scl_svd_tfidf_s,
    #p_et_count_s,
    #p_et_svd_count_s,
    #p_et_scl_svd_count_s,
    p_xgb_svd_tfidf_s,
    p_xgb_scl_svd_tfidf_s,
    p_xgb_svd_count_s,
    p_xgb_scl_svd_count_s,
    p_lr_tfidf_l,
    p_lr_svd_tfidf_l,
    p_lr_scl_svd_tfidf_l,
    p_lr_count_l,
    p_lr_svd_count_l,
    p_lr_scl_svd_count_l,
    p_nb_tfidf_l,
    p_nb_count_l,
    #p_rf_tfidf_l,
    p_rf_svd_tfidf_l,
    p_rf_scl_svd_tfidf_l,
    #p_rf_count_l,
    p_rf_svd_count_l,
    p_rf_scl_svd_count_l,
    #p_et_tfidf_l,
    #p_et_svd_tfidf_l,
    #p_et_scl_svd_tfidf_l,
    #p_et_count_l,
    #p_et_svd_count_l,
    #p_et_scl_svd_count_l,
    p_xgb_svd_tfidf_l,
    p_xgb_scl_svd_tfidf_l,
    p_xgb_svd_count_l,
    p_xgb_scl_svd_count_l,
    p_lr_tfidf_c,
    p_lr_svd_tfidf_c,
    p_lr_scl_svd_tfidf_c,
    p_lr_count_c,
    p_lr_svd_count_c,
    p_lr_scl_svd_count_c,
    p_nb_tfidf_c,
    #p_nb_count_c,
    #p_rf_tfidf_c,
    #p_rf_svd_tfidf_c,
    #p_rf_scl_svd_tfidf_c,
    #p_rf_count_c,
    #p_rf_svd_count_c,
    #p_rf_scl_svd_count_c,
    #p_et_tfidf_c,
    #p_et_svd_tfidf_c,
    #p_et_scl_svd_tfidf_c,
    #p_et_count_c,
    #p_et_svd_count_c,
    #p_et_scl_svd_count_c,
    p_xgb_svd_tfidf_c,
    p_xgb_scl_svd_tfidf_c,
    p_xgb_svd_count_c,
    p_xgb_scl_svd_count_c,
    p_lr_markov,
    p_rf_markov,
    p_et_markov,
    p_xgb_markov,
    p_lr_w2v_sent2v,
    p_lr_scl_w2v_sent2v,
    p_lr_glv_sent2v,
    p_lr_scl_glv_sent2v,
    #p_rf_w2v_sent2v,
    #p_rf_scl_w2v_sent2v,
    #p_rf_glv_sent2v,
    #p_rf_scl_glv_sent2v,
    #p_et_w2v_sent2v,
    #p_et_scl_w2v_sent2v,
    #p_et_glv_sent2v,
    #p_et_scl_glv_sent2v,
    p_xgb_w2v_sent2v,
    p_xgb_scl_w2v_sent2v,
    p_xgb_glv_sent2v,
    p_xgb_scl_glv_sent2v,
    p_nn,
    p_w2v_sent2v_nn,
    p_glv_sent2v_nn,
    p_fasttext,
    p_w2v_lstm,
    p_glv_lstm,
    p_w2v_bi_lstm,
    p_glv_bi_lstm,
    p_w2v_gru,
    p_glv_gru,
    p_w2v_ex_fasttext,
    p_glv_ex_fasttext,
    #p_lr_big_intersections,
    #p_lr_trig_intersections,
    #p_nb_big_intersections,
    #p_nb_trig_intersections,
    #p_et_big_intersections,
    #p_et_trig_intersections,
    #p_lr_senti,
    #p_lr_nltk_senti,
    #p_rf_senti,
    #p_rf_nltk_senti,
    #p_et_senti,
    #p_et_nltk_senti
)

f_all_features = f_all_nne_features + (
    f_svd_tfidf, f_scl_svd_tfidf, f_svd_count, f_scl_svd_count, f_svd_tfidf_s,
    f_scl_svd_tfidf_s, f_svd_count_s, f_scl_svd_count_s, f_svd_tfidf_l,
    f_scl_svd_tfidf_l, f_svd_count_l, f_scl_svd_count_l, f_svd_tfidf_c,
    f_scl_svd_tfidf_c, f_svd_count_c, f_scl_svd_count_c, f_markov_orders,
    f_w2v_sent2v, f_glv_sent2v)

print('all:')
p_xgb_all = apply_model(
    xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
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

p_xgb_all_m5 = apply_model(
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

p_xgb_all_200_m5 = apply_model(
    xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
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

p_xgb_all_200 = apply_model(
    xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=6,
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


make_submission(p_xgb_all, '../result', "p_xgb_all-20171214-04.csv")
make_submission(p_xgb_all_m5, '../result', "p_xgb_all-20171214-04_m5.csv")
make_submission(p_xgb_all_200_m5, '../result',
                "p_xgb_all-20171214-04_200_m5.csv")
make_submission(p_xgb_all_200, '../result', "p_xgb_all-20171214-04_200.csv")
