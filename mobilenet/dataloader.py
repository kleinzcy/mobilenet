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


if __name__ == '__main__':
    x,y = loader_train()
    print(x.shape, y.shape)
    print(x.max())
