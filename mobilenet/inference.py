#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 23:35
# @Author  : chuyu zhang
# @File    : inference.py
# @Software: PyCharm

import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
from dataloader import loader_train, loader_test, submit, dog_and_cat
from lenet import model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
# from keras import backend as K

class MobileNet:
    """
    mobilenet to complete classify task
    method: __inference, train, eval, predict
    """
    def __init__(self, classes):
        self.model = None
        self.classes = classes
        self.train_flag = True

    def __inference(self):
        """
        forward
        :param:num_classes the number of class you want to classify
        :return: model without compile
        """
        base_model = MobileNetV2(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        # print(x.get_shape())
        x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        # and a softmax logistic layer
        predictions = Dense(self.classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # frozen the weights of base model
        # for layer in base_model.layers:
        #   layer.trainable = False

        return model

    def train(self, x_train, y_train, batch_size=32, epochs=100, train=True):
        """
        pre-train or train from scartch the model
        :param x_train: input tensor,shape[num,228,228,3]
        :param y_train: one hot encoding
        :param batch_size:default 32
        :param epochs: the number of training step
        :param train: if true, train from scratch, if false, load the model form h5 file
        :return: None
        """
        if train:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
            model = self.__inference()
            optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                        patience=3,
                                                        verbose=1,
                                                        factor=0.5,
                                                        min_lr=1e-8)

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_val, y_val),
                      verbose=2, callbacks=[learning_rate_reduction])

            self.model  = model
            model.save('mobilenet.h5')
        else :
            model = load_model('mobilenet.h5')
            self.model = model

    def eval(self, x_eval, y_eval):
        model = self.model
        accuracy = model.evaluate(x_eval, y_eval, verbose=1)

        return accuracy

    def predict(self,  x_test):
        model = self.model
        y_test = model.predict(x_test)
        y_test = np.argmax(y_test)

        return y_test


def preprocess(x_train, x_test):
    """
    preprocess data to fit the input of model
    :param x: mnist train data
    :param y: the true label
    :return: processed data
    """
    print("**** preprocess data ****")
    _x_train = np.zeros((x_train.shape[0], 28, 28, 3))
    _x_test = np.zeros((x_test.shape[0], 28, 28, 3))

    for index, img in enumerate(x_train):
        # print(img)
        _x_train[index, :, :, 0] = img.reshape(28, 28)
        _x_train[index, :, :, 1] = img.reshape(28, 28)
        _x_train[index, :, :, 2] = img.reshape(28, 28)

    for index, img in enumerate(x_test):
        _x_test[index, :, :, 0] = img.reshape(28, 28)
        _x_test[index, :, :, 1] = img.reshape(28, 28)
        _x_test[index, :, :, 2] = img.reshape(28, 28)

    print(_x_train.shape)
    print(_x_test.shape)
    print("**** complete ****")

    return _x_train, _x_test


if __name__=='__main__':
    x_train, y_train, x_test, y_test = dog_and_cat()
    # print(x_train.shape, y_train.shape)

    # create model instance and train
    mobile = MobileNet(2)
    mobile.train(x_train, y_train, train=False)
    
    # eval
    acc = mobile.eval(x_test, y_test)
    print('the test loss is {}, the test accuracy is {}'.format(acc[0], acc[1]))
    print('complete...')


