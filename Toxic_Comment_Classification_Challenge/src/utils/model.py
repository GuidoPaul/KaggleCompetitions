#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from utils.ind_rnn import IndRNN

from keras.engine.topology import Input, Layer
from keras.layers.core import Dense, Flatten, Activation, SpatialDropout1D, Dropout, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.cudnn_recurrent import CuDNNLSTM, CuDNNGRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import initializers
from keras import backend as K
from keras import constraints
from keras import regularizers
from keras import optimizers as k_opt


class BaseModel:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        return NotImplemented

    def get_optimizer(self, lr, optim_name):
        optimizer = None
        if optim_name == 'nadam':
            optimizer = k_opt.Nadam(
                lr=lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                schedule_decay=0.004,
                clipvalue=1,
                clipnorm=1)
        elif optim_name == 'sgd':
            optimizer = k_opt.SGD(lr=lr, clipvalue=1, clipnorm=1)
        elif optim_name == 'rms':
            optimizer = k_opt.RMSprop(lr=lr, clipvalue=1, clipnorm=1)
        elif optim_name == 'adam':
            optimizer = k_opt.Adam(lr=lr, clipvalue=1, clipnorm=1)
        return optimizer


class AttLayer(Layer):
    def __init__(self,
                 init='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.supports_masking = True
        self.kernel_initializer = initializers.get(init)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            (input_shape[-1], 1),
            initializer=self.kernel_initializer,
            name='{}_W'.format(self.name),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.b = self.add_weight(
            (input_shape[1], ),
            initializer='zero',
            name='{}_b'.format(self.name),
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        self.u = self.add_weight(
            (input_shape[1], ),
            initializer=self.kernel_initializer,
            name='{}_u'.format(self.name),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)  # (x, 40, 1)
        uit = K.squeeze(uit, -1)  # (x, 40)
        uit = uit + self.b  # (x, 40) + (40,)
        uit = K.tanh(uit)  # (x, 40)

        ait = uit * self.u  # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait)  # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx())  # (x, 40)
            ait = mask * ait  # (x, 40) * (x, 40, )

        ait /= K.cast(
            K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


class Capsule(Layer):
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 kernel_size=(9, 1),
                 share_weights=True,
                 activation='default',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                # shape=self.kernel_size,
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (
            batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(
            u_hat_vecs[:, :, :,
                       0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (
                0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class AttenModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=256,
                 dense_size=50,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'Attention_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        x = Bidirectional(CuDNNGRU(data.embed_dim, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(data.embed_dim, return_sequences=True))(x)
        attention = AttLayer()(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool, attention])
        x = Dense(self.dense_size, activation="relu")(conc)
        outputs = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''Attention model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class CapsuleModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=256,
                 dense_size=50,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'Capsule_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        capsule = Capsule(num_capsule=128, dim_capsule=32, routings=3)(x)
        capsule = Capsule(num_capsule=64, dim_capsule=32, routings=3)(capsule)
        capsule = Capsule(num_capsule=32, dim_capsule=32, routings=3)(capsule)
        capsule = Capsule(num_capsule=16, dim_capsule=32, routings=3)(capsule)
        capsule = Capsule(num_capsule=8, dim_capsule=32, routings=3)(capsule)
        flat = Flatten()(capsule)
        outputs = Dense(6, activation="sigmoid")(flat)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''Capsule model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class CapsLSTMModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=256,
                 dense_size=32,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'CapsLSTM_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        x = Bidirectional(CuDNNLSTM(100, return_sequences=True))(x)
        capsule = Capsule(num_capsule=32, dim_capsule=32, routings=3)(x)
        flat = Flatten()(capsule)
        x = Dense(self.dense_size, activation="relu")(flat)
        outputs = Dense(6, activation="sigmoid")(flat)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''CapsLSTM model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class CapsGRUModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=256,
                 dense_size=32,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'CapsGRU_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
        capsule = Capsule(num_capsule=32, dim_capsule=32, routings=3)(x)
        flat = Flatten()(capsule)
        x = Dense(self.dense_size, activation="relu")(flat)
        outputs = Dense(6, activation="sigmoid")(flat)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''CapsGRU model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class CNNModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=128,
                 dense_size=64,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'CNN_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ), dtype='int32')
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        # kernel_size 1, 2, 3, 4, 5
        cnn1 = Conv1D(256, 1, padding='same', activation='relu')(x)
        cnn1 = Conv1D(256, 1, padding='same', activation='relu')(x)
        cnn1_max = GlobalMaxPooling1D()(cnn1)
        cnn1_avg = GlobalAveragePooling1D()(cnn1)
        cnn2 = Conv1D(256, 2, padding='same', activation='relu')(x)
        cnn2 = Conv1D(256, 2, padding='same', activation='relu')(x)
        cnn2_max = GlobalMaxPooling1D()(cnn2)
        cnn2_avg = GlobalAveragePooling1D()(cnn2)
        cnn3 = Conv1D(2, 3, padding='same', activation='relu')(x)
        cnn3 = Conv1D(2, 3, padding='same', activation='relu')(x)
        cnn3_max = GlobalMaxPooling1D()(cnn3)
        cnn3_avg = GlobalAveragePooling1D()(cnn3)
        cnn4 = Conv1D(2, 4, padding='same', activation='relu')(x)
        cnn4 = Conv1D(2, 4, padding='same', activation='relu')(x)
        cnn4_max = GlobalMaxPooling1D()(cnn4)
        cnn4_avg = GlobalAveragePooling1D()(cnn4)
        cnn5 = Conv1D(2, 5, padding='same', activation='relu')(x)
        cnn5 = Conv1D(2, 5, padding='same', activation='relu')(x)
        cnn5_max = GlobalMaxPooling1D()(cnn5)
        cnn5_avg = GlobalAveragePooling1D()(cnn5)
        cnn = concatenate(
            [
                cnn1_max, cnn1_avg, cnn2_max, cnn2_avg, cnn3_max, cnn3_avg,
                cnn4_max, cnn4_avg, cnn5_max, cnn5_avg
            ],
            axis=-1)
        x = Dense(self.dense_size, activation='relu')(cnn)
        outputs = Dense(6, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''CNN model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class DPCNNModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.0015,
                 optim_name='nadam',
                 batch_size=128,
                 dense_size=256,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'DPCNN_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ), dtype='int32')
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        embeddings = SpatialDropout1D(self.dropout)(embeddings)

        filter_nr = data.embed_dim

        kernel_size = 3
        repeat_block = 3

        reg_convo = 0.00001
        reg_dense = 0.00001
        dropout_convo = 0.0
        dropout_dense = 0.0

        def _convolutional_block(filter_nr, kernel_size, dropout, reg):
            def f(x):
                x = Conv1D(
                    filter_nr,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same',
                    kernel_regularizer=regularizers.l2(reg))(x)
                x = Dropout(dropout)(x)
                return x

            return f

        x = _convolutional_block(filter_nr, kernel_size, dropout_convo,
                                 reg_convo)(embeddings)
        x = _convolutional_block(filter_nr, kernel_size, dropout_convo,
                                 reg_convo)(x)
        x = add([embeddings, x])

        for _ in range(repeat_block):
            x = MaxPooling1D(pool_size=3, strides=2)(x)
            main = _convolutional_block(filter_nr, kernel_size, dropout_convo,
                                        reg_convo)(x)
            main = _convolutional_block(filter_nr, kernel_size, dropout_convo,
                                        reg_convo)(main)
            x = add([main, x])

        ave_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([ave_pool, max_pool])
        x = Dense(
            self.dense_size,
            activation='relu',
            kernel_regularizer=regularizers.l2(reg_dense))(conc)
        x = Dropout(dropout_dense)(x)  # TODO
        outputs = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''DPCNN model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size : {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class FastTextModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.003,
                 optim_name='nadam',
                 batch_size=128,
                 dense_size=64):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.build_model()
        self.model_description()
        self.description = 'FastText_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        ave_pool = GlobalAveragePooling1D()(embeddings)
        max_pool = GlobalMaxPooling1D()(embeddings)
        conc = concatenate([ave_pool, max_pool])
        x = Dense(self.dense_size, activation="relu")(conc)
        outputs = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''FastText model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}'''

        print(model_description)
        print(self.model.summary())


class GRUModel(BaseModel):
    def __init__(
            self,
            data,
            lr=0.001,
            # optim_name='rms',
            optim_name='nadam',
            batch_size=128,  # 256
            dense_size=64,  # 32
            dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'GRU_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        # x = Bidirectional(CuDNNGRU(data.embed_dim, return_sequences=True))(x)
        # x = Bidirectional(CuDNNGRU(data.embed_dim, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
        ave_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([ave_pool, max_pool])
        x = Dense(self.dense_size, activation="relu")(conc)
        outputs = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''GRU model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class HANModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=128,
                 dense_size=64,
                 dropout=0.5,
                 max_sent=5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.max_sent = max_sent
        self.build_model()
        self.model_description()
        self.description = 'HAN_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
        # x = Bidirectional(CuDNNGRU(data.embed_dim, return_sequences=True))(x)
        x = TimeDistributed(Dense(100, activation="relu"))(x)
        attention = AttLayer()(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool, attention])
        encoder = Model(inputs=inputs, outputs=conc)

        inputs2 = Input(shape=(self.max_sent, data.seq_length, ))
        x2 = TimeDistributed(encoder)(inputs2)
        x2 = Bidirectional(CuDNNGRU(100, return_sequences=True))(x2)
        x2 = Bidirectional(CuDNNGRU(100, return_sequences=True))(x2)
        # x2 = Bidirectional(CuDNNGRU(data.embed_dim, return_sequences=True))(x2)
        x2 = TimeDistributed(Dense(100, activation="relu"))(x2)
        attention2 = AttLayer()(x2)
        avg_pool2 = GlobalAveragePooling1D()(x2)
        max_pool2 = GlobalMaxPooling1D()(x2)
        conc2 = concatenate([avg_pool2, max_pool2, attention2])
        x2 = Dense(self.dense_size, activation="relu")(conc2)
        outputs2 = Dense(6, activation="sigmoid")(x2)
        model = Model(inputs=inputs2, outputs=outputs2)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''HAN model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class InceptionModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=128,
                 dense_size=64,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'Inception_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)

        x = Conv1D(32, 3, padding='same')(x)
        x = Conv1D(32, 3, padding='same')(x)
        x = Conv1D(64, 3, padding='same')(x)
        x = MaxPooling1D(pool_size=3, strides=2)(x)
        x = Conv1D(80, 1, padding='same')(x)
        x = Conv1D(192, 3, padding='same')(x)
        x = MaxPooling1D(pool_size=3, strides=2)(x)

        cnn1 = Conv1D(128, 1, padding='same')(x)

        cnn2 = Conv1D(256, 1, padding='same')(x)
        cnn2 = BatchNormalization()(cnn2)
        cnn2 = Activation('relu')(cnn2)
        cnn2 = Conv1D(128, 3, padding='same')(cnn2)

        cnn3 = Conv1D(256, 3, padding='same')(x)
        cnn3 = BatchNormalization()(cnn3)
        cnn3 = Activation('relu')(cnn3)
        cnn3 = Conv1D(128, 5, padding='same')(cnn3)

        cnn4 = Conv1D(128, 3, padding='same')(x)

        x = concatenate([cnn1, cnn2, cnn3, cnn4], axis=-1)

        cnn1 = Conv1D(64, 1, padding='same')(x)
        cnn1_max = GlobalMaxPooling1D()(cnn1)
        cnn1_avg = GlobalAveragePooling1D()(cnn1)

        cnn2 = Conv1D(128, 1, padding='same')(x)
        cnn2 = BatchNormalization()(cnn2)
        cnn2 = Activation('relu')(cnn2)
        cnn2 = Conv1D(64, 3, padding='same')(cnn2)
        cnn2_max = GlobalMaxPooling1D()(cnn2)
        cnn2_avg = GlobalAveragePooling1D()(cnn2)

        cnn3 = Conv1D(128, 3, padding='same')(x)
        cnn3 = BatchNormalization()(cnn3)
        cnn3 = Activation('relu')(cnn3)
        cnn3 = Conv1D(64, 5, padding='same')(cnn3)
        cnn3_max = GlobalMaxPooling1D()(cnn3)
        cnn3_avg = GlobalAveragePooling1D()(cnn3)

        cnn4 = Conv1D(64, 3, padding='same')(x)
        cnn4_max = GlobalMaxPooling1D()(cnn4)
        cnn4_avg = GlobalAveragePooling1D()(cnn4)

        inception = concatenate(
            [
                cnn1_max, cnn1_avg, cnn2_max, cnn2_avg, cnn3_max, cnn3_avg,
                cnn4_max, cnn4_avg
            ],
            axis=-1)

        # flat = Flatten()(inception)
        x = Dense(self.dense_size, activation="relu")(inception)
        outputs = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''Inception model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class IndRNNModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=128,
                 dense_size=32,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'IndRNN_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        x = IndRNN(256, return_sequences=True)(x)
        x = IndRNN(256, return_sequences=True)(x)
        x = IndRNN(128, return_sequences=True)(x)
        x = IndRNN(128, return_sequences=False)(x)
        x = Dense(self.dense_size, activation="relu")(x)
        outputs = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''IndRNN model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class LSTMModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.001,
                 optim_name='nadam',
                 batch_size=128,
                 dense_size=64,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'LSTM_Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length, ))
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)
        x = Bidirectional(CuDNNLSTM(data.embed_dim, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(data.embed_dim, return_sequences=True))(x)
        ave_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([ave_pool, max_pool])
        # bn = BatchNormalization()(pool)
        x = Dense(self.dense_size, activation="relu")(conc)
        # drop = Dropout(self.dropout)(x)
        outputs = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''LSTM model
                            lr: {self.lr}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}
                            dense_size: {self.dense_size}
                            dropout: {self.dropout}'''

        print(model_description)
        print(self.model.summary())


class RCNNModel(BaseModel):
    def __init__(self,
                 data,
                 lr=0.0005,
                 optim_name='nadam',
                 batch_size=256,
                 dense_size=100,
                 dropout=0.5):
        super().__init__(data, batch_size)
        self.lr = lr
        self.optim_name = optim_name
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.model_description()
        self.description = 'RCNN_Model'

    def build_model(self):
        data = self.data
        maxlen = data.seq_length
        inputs = Input(shape=(data.seq_length, ), dtype="int32")
        embeddings = Embedding(
            data.max_feature,
            data.embed_dim,
            weights=[data.embed_matrix],
            trainable=False)(inputs)
        x = SpatialDropout1D(self.dropout)(embeddings)

        def k_slice(x, start, end):
            return x[:, start:end]

        left_1 = Lambda(
            k_slice, arguments={'start': maxlen - 1,
                                'end': maxlen})(x)
        left_2 = Lambda(k_slice, arguments={'start': 0, 'end': maxlen - 1})(x)
        l_embedding = concatenate([left_1, left_2], axis=1)

        right_1 = Lambda(k_slice, arguments={'start': 1, 'end': maxlen})(x)
        right_2 = Lambda(k_slice, arguments={'start': 0, 'end': 1})(x)
        r_embedding = concatenate([right_1, right_2], axis=1)

        forward = CuDNNGRU(data.embed_dim, return_sequences=True)(l_embedding)
        backward = CuDNNGRU(
            data.embed_dim, return_sequences=True,
            go_backwards=True)(r_embedding)

        together = concatenate([forward, x, backward], axis=2)
        semantic = TimeDistributed(
            Dense(self.dense_size, activation="tanh"))(together)
        max_pool = GlobalMaxPooling1D()(semantic)
        ave_pool = GlobalAveragePooling1D()(semantic)
        conc = concatenate([ave_pool, max_pool])
        outputs = Dense(6, activation="sigmoid")(conc)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = self.get_optimizer(self.lr, self.optim_name)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.model = model

    def model_description(self):
        model_description = f'''RCNN model
                            dense_size: {self.dense_size}
                            lr: {self.lr}
                            dropout: {self.dropout}
                            optim_name: {self.optim_name}
                            batch_size: {self.batch_size}'''

        print(model_description)
        print(self.model.summary())


if __name__ == '__main__':

    class Data:
        def __init__(self):
            self.seq_length = 500
            self.max_feature = 170000
            self.embed_dim = 300
            self.embed_matrix = np.ones((self.max_feature, self.embed_dim))

    data = Data()
    model = HANModel(data)
