#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import preprocess_input as xception_preprocessor
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor


def read_data(img_path, labels_path, use_top_16, usage='Training'):
    assert usage in ('Training', 'Testing')

    train_df = pd.read_csv(labels_path)
    train_df["image_path"] = train_df.apply(
        lambda x: os.path.join(img_path, x["id"] + ".jpg"), axis=1)

    # using top 16 classes
    if use_top_16:
        top_breeds = sorted(
            list(train_df['breed'].value_counts().head(16).index))
        train_df = train_df[train_df['breed'].isin(top_breeds)]

    if usage is 'Training':
        print(train_df.info())
        print(train_df.head())
        print("Number of breeds of dogs: ", len(set(train_df['breed'])))

    train_img = np.array([
        image.img_to_array(image.load_img(img, target_size=(224, 224)))
        for img in train_df['image_path'].values.tolist()
    ]).astype("float32")

    train_labels = pd.get_dummies(
        train_df['breed'].reset_index(drop=True),
        columns=top_breeds).as_matrix()

    return train_img, train_labels


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
        "input_shape": (224, 224, 3),
        "pooling": "avg"
    },
    "Xception": {
        "model": Xception,
        "preprocessor": xception_preprocessor,
        "input_shape": (224, 224, 3),
        "pooling": "avg"
    }
}

batch_size = 32
epochs = 30
# num_classes = len(top_breeds)
num_classes = 16
SEED = 2018


def generate_features(train_img, train_labels):
    def get_bottleneck_features(model_info, data, labels, datagen):
        print("generating features...")
        datagen.preprocessing_function = model_info["preprocessor"]
        generator = datagen.flow(
            data, labels, shuffle=False, batch_size=batch_size, seed=SEED)
        bottleneck_model = model_info["model"](
            weights='imagenet',
            include_top=False,
            input_shape=model_info["input_shape"],
            pooling=model_info["pooling"])
        return bottleneck_model.predict_generator(
            generator, steps=(len(data) // batch_size) + 1)

    for model_name, model in models.items():
        print("Generater model feature: {}".format(model_name))
        filepath = os.path.join("../model", model_name + '_features.npy')
        if os.path.exists(filepath):
            continue
        else:
            datagen = ImageDataGenerator(
                zoom_range=0.3, width_shift_range=0.1, height_shift_range=0.1)
            features = get_bottleneck_features(model, train_img, train_labels,
                                               datagen)
            np.save(filepath, features)

        print(features.shape, train_labels.shape)


def run_train(train_labels):
    features = np.hstack([
        np.load(os.path.join('../model', model_name + '_features.npy'))
        for model_name, model in models.items()
    ])

    X_train, X_valid, y_train, y_valid = train_test_split(
        features, train_labels, test_size=0.2)

    start_time = time.time()
    logreg = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', random_state=SEED)
    logreg.fit(X_train, (y_train * range(num_classes)).sum(axis=1))

    predict_probs = logreg.predict_proba(X_valid)

    print('ensemble of features va logLoss : {}'.format(
        log_loss(y_valid, predict_probs)))
    end_time = time.time()
    print('Training time : {} {}'.format(
        np.round((end_time - start_time) / 60, 2), ' minutes'))


def main():
    train_img, train_labels = read_data("../input/train",
                                        "../input/labels.csv", True)

    # test_img, test_labels = read_data("../input/test",
    #                                   "../input/sample_submission.csv", False)

    # show_data(train_img, train_labels)

    generate_features(train_img, train_labels)

    run_train(train_labels)


if __name__ == "__main__":
    main()
