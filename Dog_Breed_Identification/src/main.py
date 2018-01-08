#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import preprocess_input as xception_preprocessor
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor


def read_data(labels_path, img_path, use_top_16):
    train_df = pd.read_csv(labels_path)

    train_df["image_path"] = train_df.apply(
        lambda x: os.path.join("../input/train", x["id"] + ".jpg"), axis=1)

    # using top 16 classes
    if use_top_16:
        top_breeds = sorted(
            list(train_df['breed'].value_counts().head(16).index))
        train_df = train_df[train_df['breed'].isin(top_breeds)]

    print(train_df.info())
    print(train_df.head())
    print("Number of breeds of dogs: ", len(set(train_df['breed'])))

    train_img = np.array([
        image.img_to_array(image.load_img(img, target_size=(224, 224)))
        for img in train_df['image_path'].values.tolist()
    ]).astype("float32")

    train_labels = np.array(train_df['breed'])

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


def generate_features(train_img, train_labels):
    batch_size = 32
    epochs = 30
    # num_classes = len(top_breeds)
    num_classes = 16
    SEED = 2018

    def get_features(model_info, data, labels, datagen):
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

    for model_name, model in models.items():
        print("Predicting : {}".format(model_name))
        filepath = os.path.join("../model", model_name + '_features.npy')
        if os.path.exists(filepath):
            features = np.load(filepath)
        else:
            datagen = ImageDataGenerator(
                zoom_range=0.3, width_shift_range=0.1, height_shift_range=0.1)
            features = get_features(model, train_img, train_labels, datagen)
            np.save(filepath, features)

        # Now that we have created or loaded the features  we need to do some predictions.
        start_time = time.time()

        print(features.shape, train_labels.shape)

        # logreg = LogisticRegression(
        #     multi_class='multinomial', solver='lbfgs', random_state=SEED)
        # logreg.fit(features, (train_labels * range(num_classes)).sum(axis=1))

        # model["predict_proba"] = logreg.predict_proba(validation_features)
        end_time = time.time()
        print('Training time : {} {}'.format(
            np.round((end_time - start_time) / 60, 2), ' minutes'))


def main():
    train_img, train_labels = read_data('../input/labels.csv',
                                        '../input/train', True)

    # show_data(train_img, train_labels)

    generate_features(train_img, train_labels)


if __name__ == "__main__":
    main()
