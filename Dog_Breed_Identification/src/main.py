#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from keras.preprocessing import image


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


def main():
    train_img, train_labels = read_data('../input/labels.csv',
                                        '../input/train', True)

    show_data(train_img, train_labels)


if __name__ == "__main__":
    main()
