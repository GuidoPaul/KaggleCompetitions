#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras import layers, models, regularizers, optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout


def preprocessing(labels_path, for_dir, usage):
    assert usage in ("Training", "Testing")
    data_df = pd.read_csv(labels_path)

    if os.path.exists(for_dir):
        shutil.rmtree(for_dir)

    if usage is "Training":
        for _, (fname, breed) in data_df.iterrows():
            for_path = '%s/%s' % (for_dir, breed)
            if not os.path.exists(for_path):
                os.makedirs(for_path)
            os.symlink('../../../input/train/%s.jpg' % fname,
                       '%s/%s.jpg' % (for_path, fname))
    else:
        breed = 0
        for fname in data_df['id']:
            for_path = '%s/%s' % (for_dir, breed)
            if not os.path.exists(for_path):
                os.makedirs(for_path)
            os.symlink('../../../input/test/%s.jpg' % fname,
                       '%s/%s.jpg' % (for_path, fname))


def data_augmentation(img_dir, img_size, batch_size, usage):
    assert usage in ("Training", "Testing")
    if usage is "Training":
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            shear_range=0.1,
            zoom_range=0.1)
        generator = datagen.flow_from_directory(
            img_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical')
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

        generator = datagen.flow_from_directory(
            img_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
    total_image_count = generator.samples
    # class_count = generator.num_class
    print(total_image_count)
    return generator


def run_train(train_generator, valid_generator, img_size, batch_size):
    conv_base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3))
    conv_base.trainable = False
    features = conv_base.predict_generator(
        train_generator, steps=int(train_generator.samples / batch_size))
    # features = conv_base.predict_generator(train_generator, steps=1)
    # print(features.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features, train_labels, test_size=0.3, random_state=SEED)

    model = models.Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(120, activation='sigmoid'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.90),
        metrics=['acc'])
    model.summary()

    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=int(train_generator.samples / batch_size),
        # steps_per_epoch=int(total_train_image_count / batch_size),
        # steps_per_epoch=1,
        epochs=3,
        validation_data=valid_generator,
        # validation_steps=int(total_val_image_count / batch_size)
        validation_steps=int(valid_generator.samples / batch_size)
        # validation_steps=1
    )


def main():
    preprocessing(
        labels_path='../input/labels.csv',
        for_dir='../data_gen/for_train',
        usage='Training')
    preprocessing(
        labels_path='../input/sample_submission.csv',
        for_dir='../data_gen/for_test',
        usage='Testing')
    train_generator = data_augmentation(
        img_dir='../data_gen/for_train',
        img_size=299,
        batch_size=32,
        usage='Training')
    valid_generator = data_augmentation(
        img_dir='../data_gen/for_test',
        img_size=299,
        batch_size=32,
        usage='Training')
    class_indices = sorted(
        [[k, v] for k, v in train_generator.class_indices.items()],
        key=lambda c: c[1])
    columns = [b[0] for b in class_indices]
    print(columns[:10])
    # run_train(train_generator, valid_generator, img_size=299, batch_size=32)


if __name__ == "__main__":
    main()
