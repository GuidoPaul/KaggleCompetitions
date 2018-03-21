#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np

from sklearn.metrics import log_loss, roc_auc_score

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import load_model


def _train_model(toxic_model, batch_size, x_train, y_train, x_valid, y_valid,
                 model_dir):
    best_auc = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0
    print(f'will train new model train.size {x_train.shape[0]},'
          f' valid.size {x_valid.shape[0]}')

    hdf5_path = ''
    while True:
        toxic_model.model.fit(
            x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
        y_pred = toxic_model.model.predict(x_valid, batch_size=batch_size)

        total_loss = 0
        total_auc = 0
        for j in range(6):
            loss = log_loss(y_valid[:, j], y_pred[:, j], eps=1e-5)
            total_loss += loss
            auc = roc_auc_score(y_valid[:, j], y_pred[:, j])
            total_auc += auc

        total_loss /= 6.
        total_auc /= 6.

        print("Epoch {0} loss {1} auc {2} best_auc {3}".format(
            current_epoch, total_loss, total_auc, best_auc))
        current_epoch += 1
        if total_auc > best_auc:
            best_auc = total_auc
            best_weights = toxic_model.model.get_weights()
            best_epoch = current_epoch

            if hdf5_path != '':
                os.remove(hdf5_path)
            model_name = toxic_model.description + \
                f'_epoch_{current_epoch}_val_auc_{best_auc:.5f}.hdf5'
            hdf5_path = os.path.join(model_dir, model_name)
            toxic_model.model.save(hdf5_path)
        else:
            if current_epoch - best_epoch == 5:
                break

    toxic_model.model.set_weights(best_weights)


class RocAucEvaluation(Callback):
    def __init__(self, batch_size, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.batch_size = batch_size
        self.interval = interval
        self.x_valid, self.y_valid = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(
                self.x_valid, batch_size=self.batch_size)

            total_loss = 0
            total_auc = 0
            for j in range(6):
                loss = log_loss(self.y_valid[:, j], y_pred[:, j], eps=1e-5)
                total_loss += loss
                auc = roc_auc_score(self.y_valid[:, j], y_pred[:, j])
                total_auc += auc
            total_loss /= 6.
            total_auc /= 6.

            logs['roc_auc_val'] = total_auc
            print(f"Epoch {epoch} loss {total_loss} auc {total_auc}")


def _train_model2(toxic_model, batch_size, x_train, y_train, x_valid, y_valid,
                  model_dir):
    print(
        f'will train new model train.size {x_train.shape[0]}, valid.size {x_valid.shape[0]}'
    )

    epochs = 20
    hdf5_path = os.path.join(model_dir, toxic_model.description + '.hdf5')

    earlyStopping = EarlyStopping(
        monitor='roc_auc_val', patience=2, verbose=0, mode='max')
    model_chk = ModelCheckpoint(
        filepath=hdf5_path,
        monitor='roc_auc_val',
        save_best_only=True,
        verbose=0,
        mode='max')
    # roc_auc_eva before others in callbacks
    roc_auc_eva = RocAucEvaluation(
        batch_size, validation_data=(x_valid, y_valid), interval=1)

    toxic_model.model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        callbacks=[roc_auc_eva, earlyStopping, model_chk],
        verbose=0)
    toxic_model.model = load_model(hdf5_path)


def create_model_path():
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, str(datetime.datetime.now()))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def train(toxic_model, model_dir, valid_split=0.1):
    print(f'call train funciton')
    x = toxic_model.data.x_train
    y = toxic_model.data.y_train
    size = x.shape[0]
    split_index = int(size * valid_split)
    x_valid = x[:split_index]
    y_valid = y[:split_index]
    x_train = x[split_index:]
    y_train = y[split_index:]

    if hasattr(toxic_model.data, 'x_procs') and toxic_model.data.x_procs:
        train_list = [x_train]
        for train_arr in toxic_model.data.x_procs:
            train_list.append(train_arr[split_index:])
        x_train = np.concatenate(train_list)
        y_train = np.concatenate([y_train] *
                                 (len(toxic_model.data.x_procs) + 1))

    if toxic_model.model is None:
        raise ValueError('model not defined!')
    _train_model(toxic_model, toxic_model.batch_size, x_train, y_train,
                 x_valid, y_valid, model_dir)
    return toxic_model.model


def train_folds(toxic_model, fold_count, log_dir):
    print(f'call train_folds fold_count: {fold_count}')
    x = toxic_model.data.x_train
    y = toxic_model.data.y_train
    fold_size = len(x) // fold_count
    origin_description = toxic_model.description
    models = []
    for fold_id in range(0, fold_count):
        print(f'call train_folds fold_id: {fold_id}')
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(x)

        x_train = np.concatenate([x[:fold_start], x[fold_end:]])
        y_train = np.concatenate([y[:fold_start], y[fold_end:]])
        if hasattr(toxic_model.data, 'x_procs') and toxic_model.data.x_procs:
            train_list = [x_train]
            for train_arr in toxic_model.data.x_procs:
                train_list.append(train_arr[:fold_start])
                train_list.append(train_arr[fold_end:])
            x_train = np.concatenate(train_list)
            y_train = np.concatenate([y_train] *
                                     (len(toxic_model.data.x_procs) + 1))

        x_valid = x[fold_start:fold_end]
        y_valid = y[fold_start:fold_end]
        toxic_model.build_model()
        toxic_model.description = origin_description + f"_fold_{fold_id}"
        _train_model(toxic_model, toxic_model.batch_size, x_train, y_train,
                     x_valid, y_valid, log_dir)
        models.append(toxic_model.model)
    return models


def train_folds_2(toxic_model, fold_count, log_dir):
    print(f'call train_folds fold_count: {fold_count}')
    x = toxic_model.data.x_train
    y = toxic_model.data.y_train
    fold_size = len(x) // fold_count
    origin_description = toxic_model.description
    models = []
    for fold_id in range(0, fold_count):
        print(f'call train_folds fold_id: {fold_id}')
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(x)

        x_train = np.concatenate([x[:fold_start], x[fold_end:]])
        y_train = np.concatenate([y[:fold_start], y[fold_end:]])
        x_valid = x[fold_start:fold_end]
        y_valid = y[fold_start:fold_end]
        if hasattr(toxic_model.data, 'x_procs') and toxic_model.data.x_procs:
            train_list = [x_train]
            valid_list = [x_valid]
            for train_arr in toxic_model.data.x_procs:
                train_list.append(train_arr[:fold_start])
                train_list.append(train_arr[fold_end:])
                valid_list.append(train_arr[fold_start:fold_end])
            x_train = np.concatenate(train_list)
            y_train = np.concatenate([y_train] *
                                     (len(toxic_model.data.x_procs) + 1))
            x_valid = np.concatenate(valid_list)
            y_valid = np.concatenate([y_valid] *
                                     (len(toxic_model.data.x_procs) + 1))

        toxic_model.build_model()
        toxic_model.description = origin_description + f"_fold_{fold_id}"
        _train_model(toxic_model, toxic_model.batch_size, x_train, y_train,
                     x_valid, y_valid, log_dir)
        models.append(toxic_model.model)
    return models
