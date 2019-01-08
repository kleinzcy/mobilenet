#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 23:35
# @Author  : chuyu zhang
# @File    : inference.py
# @Software: PyCharm
import tensorflow as tf
hello = tf.constant('hello,tensorf')
sess = tf.Session()
print(sess.run(hello))