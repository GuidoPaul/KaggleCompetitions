#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from keras.models import load_model

from utils import data_source
from utils.model import AttLayer


def infer_result(models, toxic_data, result_dir, result_name='submit.csv'):
    if not isinstance(models, list):
        models = [models]
    test_predicts_list = []
    for i, toxic_model in enumerate(models):
        print(f'train {i}-th model ......')
        test_predicts = toxic_model.predict(toxic_data.x_test, batch_size=128)
        print(f'train {i}-th model finish')
        test_predicts_list.append(test_predicts)

    test_predicts = np.ones(test_predicts_list[0].shape)
    for fold_predict in test_predicts_list:
        test_predicts *= fold_predict

    test_predicts **= (1. / len(test_predicts_list))

    test_ids = toxic_data.test_df["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    CLASSES = data_source.CLASSES

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]

    output_path = os.path.join(result_dir, result_name)
    test_predicts.to_csv(output_path, index=False)


EMBED_FILES = {
    'crawl': '../model/crawl-300d-2M.vec',
    'wiki': '../model/wiki.en.bin',
    'glove': '../model/glove.840B.300d.txt'
}

if __name__ == '__main__':
    model_path_list = []

    model_list = [
        load_model(path, custom_objects={'AttLayer': AttLayer})
        for path in model_path_list
    ]
    result_path = ''

    toxic_data = data_source.DataSource(EMBED_FILES, embed_flag='crawl')

    infer_result(model_list, toxic_data, result_path, 'attention.csv')
