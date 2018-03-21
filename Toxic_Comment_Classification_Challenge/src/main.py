#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import data_source
from utils import infer
from utils import log
from utils import model
from utils import train

EMBED_FILES = {
    'crawl': '../model/crawl-300d-2M.vec',
    'wiki': '../model/wiki.en.bin',
    'glove': '../model/glove.840B.300d.txt'
}

log_dir = log.init_log()
train_fold = True


def main():
    # if han sent_flag=True
    toxic_data = data_source.DataSource(
        EMBED_FILES, embed_flag='crawl', sent_flag=False)
    print(toxic_data.description())

    train_model = model.IndRNNModel(toxic_data)

    if train_fold:
        result_model = train.train_folds(train_model, 10, log_dir)
    else:
        result_model = train.train(train_model, log_dir)
    print('train finish')

    result_file = train_model.description + '.csv'
    print(f'result_file: {result_file}')
    infer.infer_result(result_model, toxic_data, log_dir, result_file)


if __name__ == "__main__":
    main()
