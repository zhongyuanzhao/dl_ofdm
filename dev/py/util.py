#!/usr/bin/env python
######################################################################################
# Library for Utility Functions in Wireless Communication
# Author: Zhongyuan Zhao 
# Date: 2018-07-05
# Link: https://github.com/zhongyuanzhao/dl_ofdm
# Cite this work:
# Zhongyuan Zhao, Mehmet C. Vuran, Fujuan Guo, and Stephen Scott, "Deep-Waveform: 
# A Learned OFDM Receiver Based on Deep Complex-valued Convolutional Networks," 
# EESS.SP, vol abs/1810.07181, Mar. 2021, [Online] https://arxiv.org/abs/1810.07181
# 
# Copyright (c) 2021: Zhongyuan Zhao
# Houston, Texas, United States
# <zhongyuan.zhao@huskers.unl.edu>
######################################################################################


import tensorflow as tf
import numpy as np
import os
# from sklearn.preprocessing import OneHotEncoder
from model import *


def bit_source(nbits, frame_size, msg_length):
    '''
    Generate uniform random m_order one hot symbols
    :param frame_size: Equivalent to FFT size in OFDM
    :param msg_length: number of frames
    :return: bits
    '''
    # nbits = int(np.log2(m_order))
    bits = np.random.randint(0, 2, (int(msg_length), int(frame_size), int(nbits)))
    return bits


def ber_calc(conf_matrix):
    assert(conf_matrix.shape==(2,2))
    totalbits = np.sum(conf_matrix)
    errorbits = conf_matrix[0][1] + conf_matrix[1][0]
    return float(errorbits)/float(totalbits)


def ber_tensor(conf_matrix):
    totalbits = tf.reduce_sum(conf_matrix)
    errorbits = conf_matrix[0, 1]+conf_matrix[1, 0]
    berlinear = tf.divide(errorbits, totalbits, name="BER")
    return tf.log(berlinear), tf.cast(berlinear, tf.float32)

