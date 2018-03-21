#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from fastText import load_model

embed_file = '../model/wiki.en.bin'
ft_model = load_model(embed_file)
input_matrix = ft_model.get_input_matrix()
output_matrix = ft_model.get_output_matrix()

emb_mean, emb_std = input_matrix.mean(), input_matrix.std()
print(emb_mean)
print(emb_std)

emb_mean, emb_std = output_matrix.mean(), output_matrix.std()
print(emb_mean)
print(emb_std)


def load_crawl_embed_index(file_path):
    embed_index = {}
    with open(file_path) as f:
        for line in f.read().split("\n")[1:-1]:
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:-1], dtype='float32')
            embed_index[word] = coefs

    print('Found %s word vectors.' % len(embed_index))
    return embed_index


embed_file = '../model/crawl-300d-2M.vec'
embed_index = load_crawl_embed_index(embed_file)
all_embs = np.stack(embed_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
print(emb_mean)
print(emb_std)
