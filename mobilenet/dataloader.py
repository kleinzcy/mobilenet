#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time     : 2018/11/21 13:42
# @Author   : klein
# @File     : dataloader.py
# @Software : PyCharm

import numpy as np
import pandas as pd
from sklearn import preprocessing
import warnings
import os
import time
from PIL import Image
from tqdm import  tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def loader_train(filename=r'data/train.csv'):
    print('loading train data...')
    data = pd.read_csv(filename)
    y = data.iloc[:, 0].values
    # y = y[:1000]
    y = one_hot_encoder(y)
    x = data.iloc[:, 1:].values
    # x = x[:1000, :]
    x = normalized_data(x)
    print('complete...')
    return x, y


def loader_test(filename=r'data/test.csv'):
    print('loading test data...')
    x = pd.read_csv(filename).values
    # x = x[:100, :]
    x = normalized_data(x)
    print('complete...')
    return x


def normalized_data(data):
    """
    归一化数据
    :param data:要归一化的数据
    :return:
    """
    data = np.array(data)
    data = data/data.max()
    return data


def one_hot_encoder(label):
    """

    :param label: 二维向量，(n.1)
    :return: 编码后的lable，(n,m)
    """
    # change the shape of label
    label = np.array(label).reshape(-1, 1)

    # init one hot encoder
    enc = preprocessing.OneHotEncoder(categories='auto')

    num = np.unique(label,axis=0)
    enc.fit(num)
    new_label = enc.transform(label).toarray()

    return new_label


def one_hot_decoder(label):
    """
    decoder
    :param label: type: ndarray, shape: [n ,10]
    :return: type: ndarray, shape: [n, 1]
    """
    return np.argmax(label, axis=1)


def submit(y, filename='result.csv'):
    """
    submit result
    :param y: the result
    :param filename: filename created for the result
    :return: None
    """
    n = y.shape[0]
    id = []
    for i in range(1,n+1):
        id.append(i)
    result = pd.DataFrame({'ImageId':id, 'Label':y})
    result.to_csv(filename, index=False, header=True)


def dog_and_cat():
    """
    load the cat and dog dataset. using numpy and PIL, next time, I will try keras
    to load file directly.
    :return: numpy ndarray shape[number, height, width, 3]
    """
    path = os.path.join(os.getcwd(), 'data/cat-and-dog/train')
    path_ = os.path.join(os.getcwd(), 'data/cat-and-dog/test')

    # this a bug in load data from .npy file
    if os.path.exists(path) and os.path.exists(path_):
        start = time.time()
        print("**** load data from .npy file ****")
        x_train = np.load(os.path.join(path, 'x.npy'))
        y_train = np.load(os.path.join(path, 'y.npy'))
        x_test = np.load(os.path.join(path_, 'x.npy'))
        y_test = np.load(os.path.join(path_, 'y.npy'))
        end = time.time() - start
        print("**** data has been load, spend {} second ****".format(end))
    else:
        start = time.time()
        print("**** load data from picture, it takes a while ****")
        _path = '/data/cat-and-dog/training_set/'
        print('load train')
        x_train, y_train = _dog_and_cat(_path)
        _path = '/data/cat-and-dog/test_set/'
        x_test, y_test = _dog_and_cat(_path)
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(path_):
            os.mkdir(path_)

        # save file as .npy format, beacuse it takes a long time to load data from image
        np.save(os.path.join(path, 'x.npy'), x_train)
        np.save(os.path.join(path, 'y.npy'), y_train)
        np.save(os.path.join(path_, 'x.npy'), x_test)
        np.save(os.path.join(path_, 'y.npy'), y_test)
        end = time.time() - start
        print("**** data has been load, spend {} seconds ****".format(end))

    return x_train, y_train, x_test, y_test


def _dog_and_cat(_path):
    dir_path = os.getcwd()
    # _path = '/data/cat-and-dog/training_set/'
    path = dir_path + _path
    # x is feature, y is label
    x = []
    y = []
    for animal in os.listdir(path):
        animal = os.path.join(path,animal)
        imgs = []
        for img_path in tqdm(os.listdir(animal)):
            img_path = os.path.join(animal, img_path)
            # resize the img
            img = Image.open(img_path).resize((40, 40))
            img = np.array(img)
            imgs.append(img)
        label = [animal.split(r'/')[-1]]*len(imgs)
        y.extend(label)
        x.extend(imgs)

    y = np.array(y).reshape(-1,1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    # print(type(y))
    # y = np.array(y)
    x = np.array(x)
    # np.save(os.path.join(dir_path,'/data/cat-and-dog/training_x.npz'), x)
    return x,y


if __name__ == '__main__':
    x,y = loader_train()
    print(x.shape, y.shape)
    print(x.max())
