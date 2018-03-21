#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from nltk import tokenize

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from fastText import load_model

path = '../input/'

TRAIN_DATA_FILE = f'{path}train.csv'
TRAIN_PROCESS_FILES = [
    # f'{path}train_clean.csv',
    # f'{path}train_drop.csv',
    # f'{path}train_shuffle.csv',
    # f'{path}train_de.csv',
    # f'{path}train_es.csv',
    # f'{path}train_fr.csv',
]
TEST_DATA_FILE = f'{path}test.csv'

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]


class DataSource(object):
    def __init__(
            self,
            embed_files,
            seq_length=320,  # 320
            embed_flag='crawl',
            sent_flag=False):
        self.train_file = TRAIN_DATA_FILE
        self.process_files = TRAIN_PROCESS_FILES
        self.test_file = TEST_DATA_FILE
        self.embed_file = embed_files[embed_flag]
        self.seq_length = seq_length

        print(f'read train data: {self.train_file} '
              f'and test data: {self.test_file}')
        self.train_df = pd.read_csv(self.train_file)
        self.test_df = pd.read_csv(self.test_file)

        self.train_df["comment_text"].fillna(NAN_WORD, inplace=True)
        self.test_df["comment_text"].fillna(NAN_WORD, inplace=True)

        sentences_train = self.train_df["comment_text"].values
        sentences_test = self.test_df["comment_text"].values
        self.y_train = self.train_df[CLASSES].values

        print(f'train sentences shape: {sentences_train.shape}')
        print(f'test sentences shape: {sentences_test.shape}')
        print(f'y train shape: {self.y_train.shape}')

        sentences_all = list(sentences_train)
        sentences_procs = []
        sentences_df_procs = []
        for train_pro in self.process_files:
            df = pd.read_csv(train_pro)
            df["comment_text"].fillna(NAN_WORD, inplace=True)
            sent = df["comment_text"].values
            sentences_procs.append(sent)
            sentences_all.extend(list(sent))
            sentences_df_procs.append(df)

        print('Tokenzie sentence in train set and test set...')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences_all)

        if not sent_flag:
            tokenized_train = tokenizer.texts_to_sequences(sentences_train)
            self.x_train = pad_sequences(tokenized_train, maxlen=seq_length)

            tokenized_test = tokenizer.texts_to_sequences(sentences_test)
            self.x_test = pad_sequences(tokenized_test, maxlen=seq_length)

            self.x_procs = []
            for sent in sentences_procs:
                tokenized_procs = tokenizer.texts_to_sequences(sent)
                tokenized_procs = pad_sequences(
                    tokenized_procs, maxlen=seq_length)
                self.x_procs.append(tokenized_procs)
        else:
            sentences_train = self.train_df["comment_text"].apply(
                lambda x: tokenize.sent_tokenize(x))
            sentences_test = self.test_df["comment_text"].apply(
                lambda x: tokenize.sent_tokenize(x))

            max_sent = 5
            self.x_train = self.sentenize(tokenizer, sentences_train, max_sent,
                                          seq_length)
            self.x_test = self.sentenize(tokenizer, sentences_test, max_sent,
                                         seq_length)

            self.x_procs = []
            for sent_df in sentences_df_procs:
                sentences_df = sent_df["comment_text"].apply(
                    lambda x: tokenize.sent_tokenize(x))
                tokenized_procs = self.sentenize(tokenizer, sentences_df,
                                                 max_sent, seq_length)
                self.x_procs.append(tokenized_procs)

        words_dict = tokenizer.word_index
        self.max_feature = len(words_dict) + 1

        print(f'Loading {embed_flag} embeddings...')
        if embed_flag is 'wiki':
            ft_model = load_model(self.embed_file)
            self.embed_dim = ft_model.get_dimension()
            self.embed_matrix = self.get_wiki_embed_matrix(
                words_dict, ft_model)
        elif embed_flag is 'crawl':
            embed_index = self.load_crawl_embed_index(self.embed_file)
            self.embed_dim = list(embed_index.values())[0].shape[0]  # 300
            self.embed_matrix = self.get_crawl_or_glove_embed_matrix(
                words_dict, embed_index)
        else:
            embed_index = self.load_glove_embed_index(self.embed_file)
            self.embed_dim = list(embed_index.values())[0].shape[0]  # 300
            self.embed_matrix = self.get_crawl_or_glove_embed_matrix(
                words_dict, embed_index)

    def sentenize(self, tk, data_sentences, max_sent, seq_length):
        sent_matrix = np.zeros(
            (len(data_sentences), max_sent, seq_length), dtype="int32")
        for i, sentences in enumerate(data_sentences):
            for j, sent in enumerate(sentences):
                if j < max_sent:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        try:
                            # if k < seq_length and tk.word_index[word] < self.max_feature:
                            if k < seq_length:
                                sent_matrix[i, j, k] = tk.word_index[word]
                        except Exception:
                            sent_matrix[i, j, k] = 0
                        k = k + 1
        return sent_matrix

    def load_crawl_embed_index(self, file_path):
        embed_index = {}
        with open(file_path) as f:
            for line in f.read().split("\n")[1:-1]:
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:-1], dtype='float32')
                embed_index[word] = coefs

        print('Found %s word vectors.' % len(embed_index))
        return embed_index

    def load_glove_embed_index(self, file_path):
        embed_index = {}
        with open(file_path) as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embed_index[word] = coefs

        print('Found %s word vectors.' % len(embed_index))
        return embed_index

    def get_wiki_embed_matrix(self, words_dict, ft_model):
        input_matrix = ft_model.get_input_matrix()
        emb_mean, emb_std = input_matrix.mean(), input_matrix.std()
        embed_matrix = np.random.normal(emb_mean, emb_std, (self.max_feature,
                                                            self.embed_dim))

        for word, i in words_dict.items():
            # if i >= self.max_feature:
            #     continue
            embed_vector = ft_model.get_word_vector(word).astype('float32')
            if embed_vector is not None:
                embed_matrix[i] = embed_vector
        return embed_matrix

    def get_crawl_or_glove_embed_matrix(self, words_dict, embed_index):
        all_embs = np.stack(embed_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_matrix = np.random.normal(emb_mean, emb_std, (self.max_feature,
                                                            self.embed_dim))

        for word, i in words_dict.items():
            # if i >= self.max_feature:
            #     continue
            embed_vector = embed_index.get(word)
            if embed_vector is not None:
                embed_matrix[i] = embed_vector
        return embed_matrix

    def description(self):
        return f'''Data Source use
        embed_file: {self.embed_file}
        embed_dim: {self.embed_dim}
        seq_length: {self.seq_length}
        max_feature: {self.max_feature}
        x_train.shape: {self.x_train.shape}
        y_train.shape: {self.y_train.shape}
        x_test.shape: {self.x_test.shape}
        '''
