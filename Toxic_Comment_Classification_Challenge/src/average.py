#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sklearn.preprocessing import minmax_scale

CLASSES = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]
sample_path = '../input/sample_submission.csv'
result_path = '../result/avg_atten_gru_dpcnn_lstm_rcnn_blend_20180320-05.csv'


def make_submission(predictions, labels_path, file_path):
    df_pred = pd.read_csv(labels_path)

    for i, c in enumerate(df_pred.columns[1:]):
        df_pred[c] = predictions[:, i]

    df_pred.to_csv(file_path, index=None)


def bag_by_average(test_predicts_list):
    bagged_predicts = np.zeros(test_predicts_list[0].shape)
    for predict in test_predicts_list:
        bagged_predicts += predict

    bagged_predicts /= len(test_predicts_list)
    return bagged_predicts


def bag_by_geomean(test_predicts_list):
    bagged_predicts = np.ones(test_predicts_list[0].shape)
    for predict in test_predicts_list:
        bagged_predicts *= predict

    bagged_predicts **= (1. / len(test_predicts_list))
    return bagged_predicts


def bag_by_rankmean(test_predicts_list):
    # p4 = np.vstack([p.values for p in test_predicts_list])
    p4 = np.vstack([p for p in test_predicts_list])
    order = p4.argsort(axis=0)
    ranks = order.argsort(axis=0)
    ranks = np.divide(ranks, ranks.shape[0])
    length = test_predicts_list[0].shape[0]
    r2 = np.stack(
        [
            ranks[i * length:(i + 1) * length, :]
            for i, _ in enumerate(test_predicts_list)
        ],
        axis=2)
    bagged_ranks = np.mean(r2, axis=2)

    bagged_predicts = np.zeros(bagged_ranks.shape)
    for i in range(6):
        interp = interp1d(ranks[:, i], p4[:, i])
        bagged_predicts[:, i] = interp(bagged_ranks[:, i])
    return bagged_predicts


def average_res(files, weights=None):
    if weights and len(weights) != len(files):
        raise ValueError('weights should has same lengths with files')
    if weights is None:
        weights = [1] * len(files)

    test_predicts_list = []
    for file, weight in zip(files, weights):
        df = pd.read_csv(file)
        for label in CLASSES:
            df[label] = minmax_scale(df[label])
        test_predicts_list.append(df.loc[:, CLASSES].as_matrix())

    test_predicts = bag_by_average(test_predicts_list)

    make_submission(test_predicts, sample_path, result_path)


def main():
    # average_res([
    #     '../result/new/multi_crawl/Attention_Model_fold_9_multi_crawl_9860.csv',
    #     '../result/new/multi_crawl/CNN_Model_fold_9_multi_crawl_9845.csv',
    #     '../result/pred_models/Double_GRU_fold_9_new_crawl_9858.csv',
    #     '../result/new/crawl/DPCNN_Model_fold_9_crawl_9853_2.csv',
    #     '../result/new/crawl/LSTM_Model_fold_9_crawl_9856.csv',
    #     '../result/pred_models/RCNN_Model_fold_9_new_crawl_9855.csv',
    # ], '../result/pred_avg_atten_cnn_doublegru_dpcnn_lstm_rcnn_20180310-02.csv') # 9863

    # average_res(
    #     [
    #         '../result/new/multi_crawl/Attention_Model_fold_9_multi_crawl_9860.csv',
    #         '../result/new/multi_crawl/CNN_Model_fold_9_multi_crawl_9845.csv',
    #         '../result/new/multi_crawl/DPCNN_Model_fold_9_multi_crawl_9856.csv',
    #         '../result/new/multi_crawl/RCNN_Model_fold_9_multi_crawl_9857.csv',
    #         '../result/new/crawl/DPCNN_Model_fold_9_crawl_9853_2.csv',
    #         '../result/new/crawl/LSTM_Model_fold_9_crawl_9856.csv',
    #         '../result/pred_models/Double_GRU_fold_9_new_crawl_9858.csv',
    #         '../result/pred_models/RCNN_Model_fold_9_new_crawl_9855.csv',
    #         '../result/new/wiki/Attention_Model_fold_9_wiki_9854.csv',  # ?
    #         '../result/new/wiki/Double_GRU_fold_9_wiki_9851.csv',  # ?
    #         '../result/others/blend_it_all_9868.csv',
    #         '../result/others/one_more_blend_9865.csv',
    #         '../result/others/hight_of_blend_v2_9861.csv',
    #         '../result/others/one_more_blend_9859.csv',
    #         '../result/others/submission_9858.csv',
    #         '../result/others/corr_blend_9855.csv',
    #         '../result/others/superblend_1_9854.csv',
    #     ],
    #     '../result/avg_atten_cnn_gru_dpcnn_lstm_rcnn_blend_20180312-05.csv',
    #     [1.5] * 10 + [1] * 7)  # 9872 better

    # average_res(
    #     [
    #         '../result/new/multi_crawl/Attention_Model_fold_9_multi_crawl_9860.csv',
    #         '../result/new/multi_crawl/DPCNN_Model_fold_9_multi_crawl_9856.csv',
    #         '../result/new/multi_crawl/GRU_Model_fold_9_multi_crawl_9861.csv',  # 0317-02
    #         '../result/new/multi_crawl/LSTM_Model_fold_9_multi_crawl_9860.csv',
    #         '../result/new/multi_crawl/RCNN_Model_fold_9_multi_crawl_9857.csv',
    #         '../result/new/multi_wiki/GRU_Model_fold_9_multi_wiki_9862.csv',
    #         '../result/new/crawl/Attention_Model_fold_9_crawl_9855.csv',
    #         '../result/new/crawl/Double_GRU_fold_9_crawl_9854.csv',
    #         '../result/new/crawl/LSTM_Model_fold_9_crawl_9856.csv',
    #         '../result/pred_models/Double_GRU_fold_9_new_crawl_9858.csv',
    #         '../result/pred_models/Double_GRU_fold_9_wiki_9854.csv',
    #         '../result/pred_models/RCNN_Model_fold_9_crawl_9854.csv',
    #         '../result/pred_models/RCNN_Model_fold_9_new_crawl_9855.csv',
    #         '../result/new/wiki/Attention_Model_fold_9_wiki_9854.csv',
    #         '../result/others/blend_it_all_9868.csv',
    #         '../result/others/one_more_blend_9865.csv',
    #         '../result/others/hight_of_blend_v2_9861.csv',
    #         '../result/others/preprocessed_blend_9860.csv',  # 0316-04
    #         '../result/others/one_more_blend_9859.csv',
    #         '../result/others/submission_9858.csv',
    #         '../result/others/corr_blend_9855.csv',
    #     ],
    #     '../result/avg_atten_cnn_gru_dpcnn_lstm_rcnn_blend_20180317-02.csv')
    # # 0316-05 minmax_scale
    average_res([
        '../result/new/multi_crawl/Attention_Model_fold_9_multi_crawl_9860.csv',
        '../result/new/multi_crawl/CapsGRU_Model_fold_9_multi_crawl.csv',  #
        '../result/new/multi_crawl/CapsLSTM_Model_fold_9_multi_crawl.csv',  #
        '../result/new/multi_crawl/DPCNN_Model_fold_9_multi_crawl_9856.csv',
        '../result/new/multi_crawl/GRU_Model_fold_9_multi_crawl_9861.csv',
        '../result/new/multi_crawl/LSTM_Model_fold_9_multi_crawl_9860.csv',
        '../result/new/multi_crawl/RCNN_Model_fold_9_multi_crawl_9857.csv',
        '../result/new/multi_wiki/Attention_Model_fold_9_multi_wiki_9859.csv',
        '../result/new/multi_wiki/GRU_Model_fold_9_multi_wiki_9862.csv',
        '../result/new/multi_wiki/RCNN_Model_fold_9_multi_wiki_9855.csv',
        '../result/new/crawl/Attention_Model_fold_9_crawl_9855.csv',
        '../result/new/crawl/Double_GRU_fold_9_crawl_9854.csv',
        '../result/new/crawl/LSTM_Model_fold_9_crawl_9856.csv',
        '../result/pred_models/Double_GRU_fold_9_new_crawl_9858.csv',
        '../result/pred_models/Double_GRU_fold_9_wiki_9854.csv',
        '../result/pred_models/RCNN_Model_fold_9_crawl_9854.csv',
        '../result/pred_models/RCNN_Model_fold_9_new_crawl_9855.csv',
        '../result/new/wiki/Attention_Model_fold_9_wiki_9854.csv',
        '../result/others/blend_it_all_9868.csv',
        '../result/others/blend_it_all_9868.csv',  # 0319-04
        '../result/others/one_more_blend_9865.csv',
        '../result/others/hight_of_blend_v2_9861.csv',
        '../result/others/preprocessed_blend_9860.csv',  # 0318-04
        '../result/others/preprocessed_blend_9860.csv',  # 0319-05
        '../result/others/preprocessed_blend_9860.csv',
        '../result/others/preprocessed_blend_9860.csv',  # 0320-05
        '../result/others/one_more_blend_9859.csv',
        '../result/others/submission_9858.csv',
        '../result/others/corr_blend_9855.csv',
        '../result/others/tr8.csv',  # 0320-01
        # '../result/others/superblend_1_9854.csv',
        # '../result/others/Blended_out_2_9837.csv',
    ])


if __name__ == "__main__":
    main()
