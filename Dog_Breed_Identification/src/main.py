#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd

# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
# from sklearn.model_selection import train_test_split

# from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import preprocess_input as xception_preprocessor
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint


def read_data(img_path, labels_path, img_size, use_top_16, usage="Training"):
    assert usage in ("Training", "Testing")

    data_df = pd.read_csv(labels_path)
    data_df["image_path"] = data_df.apply(
        lambda x: os.path.join(img_path, x["id"] + ".jpg"), axis=1)

    # using top 16 classes
    if usage is "Training" and use_top_16:
        top_breeds = sorted(
            list(data_df["breed"].value_counts().head(16).index))
        data_df = data_df[data_df["breed"].isin(top_breeds)]

    data_img = np.array([
        image.img_to_array(
            image.load_img(img, target_size=(img_size, img_size)))
        for img in data_df["image_path"].values.tolist()
    ]).astype("float32")

    if usage is 'Training':
        print(data_df.info())
        print(data_df.head())
        print("Number of breeds of dogs: ", len(set(data_df["breed"])))

        # one hot labels
        data_labels = np.asarray(pd.get_dummies(data_df["breed"], sparse=True))

        return data_img, data_labels
    else:
        return data_img


def show_data(train_img, train_labels):
    import matplotlib.pyplot as plt

    for i in range(1, 5):
        plt.subplot(2, 2, i)
        rint = np.random.randint(0, len(train_img))
        plt.title(train_labels[rint])
        plt.axis('off')
        plt.imshow(train_img[rint].astype(np.uint8))

    plt.show()


models = {
    "InceptionV3": {
        "model": InceptionV3,
        "preprocessor": inception_v3_preprocessor,
        "input_shape": (299, 299, 3),
        "pooling": "avg"
    },
    "Xception": {
        "model": Xception,
        "preprocessor": xception_preprocessor,
        "input_shape": (299, 299, 3),
        "pooling": "avg"
    }
}

batch_size = 32
epochs = 30
num_classes = 120
SEED = 2018
model_path = '/tmp/nn_model.h5'


def generate_features(data_img, usage):
    assert usage in ("Training", "Testing")

    def get_bottleneck_features(model_info, data, datagen):
        print("generating features...")
        datagen.preprocessing_function = model_info["preprocessor"]
        generator = datagen.flow(
            data, shuffle=False, batch_size=batch_size, seed=SEED)
        bottleneck_model = model_info["model"](
            weights='imagenet',
            include_top=False,
            input_shape=model_info["input_shape"],
            pooling=model_info["pooling"])
        return bottleneck_model.predict_generator(
            generator, steps=(len(data) // batch_size) + 1)

    for model_name, model in models.items():
        print("Generater model feature: {}".format(model_name))
        if usage is "Training":
            filename = model_name + "_features.npy"
        else:
            filename = model_name + "_features_test.npy"
        filepath = os.path.join("../model", filename)
        if os.path.exists(filepath):
            continue
        else:
            datagen = ImageDataGenerator(
                zoom_range=0.3, width_shift_range=0.1, height_shift_range=0.1)
            features = get_bottleneck_features(model, data_img, datagen)
            np.save(filepath, features)

        print(features.shape)


def build_model():
    # inputs = Input(features.shape[1:])
    # x = inputs
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='softmax')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(120, activation='softmax')(x)
    # model = Model(inputs, x)

    inputs = Input((4096, ))
    x = inputs
    x = Dropout(0.5)(x)
    x = Dense(120, activation='softmax')(x)
    model = Model(inputs, x)

    # model = Sequential()
    # model.add(Dense(256, input_dim=4096))
    # # model.add(Dense(120, input_shape=(4096,)))
    # model.add(Dropout(0.5))
    # model.add(Dense(120))

    # model = Sequential()
    # model.add(Dense(256, input_shape=(4096, ), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(120))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.summary()

    return model


def make_submission(predictions, labels_path, submission_dir, file_name):
    file_path = os.path.join(submission_dir, os.path.basename(file_name))
    df_pred = pd.read_csv(labels_path)

    for i, c in enumerate(df_pred.columns[1:]):
        df_pred[c] = predictions[:, i]

    df_pred.to_csv(file_path, index=None)


def run_train(train_labels):
    features = np.hstack([
        np.load(os.path.join('../model', model_name + '_features.npy'))
        for model_name, model in models.items()
    ])
    features_test = np.hstack([
        np.load(os.path.join('../model', model_name + '_features_test.npy'))
        for model_name, model in models.items()
    ])
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        features, train_labels, test_size=0.3, random_state=SEED)
    """
    print(features.shape)
    print(features.shape[1:])
    print(type(features.shape[1:]))

    start_time = time.time()

    model = build_model()

    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='auto')
    model_chk = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1)

    model.fit(
        features,
        train_labels,
        batch_size=32,
        epochs=20,
        validation_split=0.3,
        callbacks=[earlyStopping, model_chk],
        verbose=1)
    model = load_model(model_path)
    predictions = model.predict(features_test, verbose=1)
    make_submission(predictions, '../input/sample_submission.csv', '../result',
                    'pred-20180112-02.csv')
    end_time = time.time()
    print('Training time : {} {}'.format(
        np.round((end_time - start_time) / 60, 2), ' minutes'))


def lr():
    # logreg = LogisticRegression(
    #     multi_class='multinomial', solver='lbfgs', random_state=SEED)
    # logreg.fit(X_train, (y_train * range(num_classes)).sum(axis=1))

    # predict_probs = logreg.predict_proba(X_valid)

    # print('ensemble of features va logLoss : {}'.format(
    #     log_loss(y_valid, predict_probs)))

    # output = logreg.predict_proba(features_test)
    pass


def main():
    train_img, train_labels = read_data(
        img_path="../input/train",
        labels_path="../input/labels.csv",
        img_size=299,
        use_top_16=True,
        usage="Training")

    test_img = read_data(
        img_path="../input/test",
        labels_path="../input/sample_submission.csv",
        img_size=299,
        use_top_16=False,
        usage="Testing")

    # show_data(train_img, train_labels)

    generate_features(train_img, usage="Training")
    generate_features(test_img, usage="Testing")

    # data_df = pd.read_csv("../input/labels.csv")
    # train_labels = np.asarray(pd.get_dummies(data_df["breed"], sparse=True))
    run_train(train_labels)


if __name__ == "__main__":
    main()
