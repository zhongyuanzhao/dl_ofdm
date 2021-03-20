#!/usr/bin/env python
######################################################################################
# Library for DCCN models and helper functions
# Author: Zhongyuan Zhao 
# Date: 2021-03-10
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
from radio import *
tf.disable_eager_execution()


EXPAND_FACTOR = 4

def load_model(path, session):
    graph = session.graph
    saver = tf.train.import_meta_graph(path + '.meta', clear_devices=True)

    saver.restore(session, path)
    print("Load Model: %s" % (path))

    x = graph.get_tensor_by_name('bits_in:0')
    iq_receiver = graph.get_tensor_by_name('input:0')
    outputs = graph.get_tensor_by_name('output:0')
    #train_op = graph.get_tensor_by_name('train_op:0')
    total_loss = graph.get_tensor_by_name('cost:0')
    ber = graph.get_tensor_by_name('log_ber:0')
    berlin = graph.get_tensor_by_name('linear_ber:0')
    conf_matrix = graph.get_tensor_by_name('conf_matrix:0')
    power_tx = graph.get_tensor_by_name('tx_power:0')
    noise_pwr = graph.get_tensor_by_name('noise_power:0')
    iq_rx = graph.get_tensor_by_name('iq_rx:0')
    iq_tx = graph.get_tensor_by_name('iq_tx:0')
    ce_mean = graph.get_tensor_by_name('ce_mean:0')
    SNR = graph.get_tensor_by_name('SNR:0')
    norm = graph.get_tensor_by_name('Norm:0')
    return x, iq_receiver, outputs, total_loss, ber, berlin, conf_matrix, power_tx, noise_pwr, iq_rx, iq_tx, ce_mean, SNR, norm


def load_model_np(path, session):
    graph = session.graph
    saver = tf.train.import_meta_graph(path + '.meta', clear_devices=True)

    saver.restore(session, path)
    print("Load Model: %s" % (path))

    y = graph.get_tensor_by_name('bits_in:0')
    x = graph.get_tensor_by_name('tx_ofdm:0')
    iq_receiver = graph.get_tensor_by_name('input:0')
    outputs = graph.get_tensor_by_name('output:0')
    total_loss = graph.get_tensor_by_name('cost:0')
    ber = graph.get_tensor_by_name('log_ber:0')
    berlin = graph.get_tensor_by_name('linear_ber:0')
    conf_matrix = graph.get_tensor_by_name('conf_matrix:0')
    power_tx = graph.get_tensor_by_name('tx_power:0')
    noise_pwr = graph.get_tensor_by_name('noise_power:0')
    iq_rx = graph.get_tensor_by_name('iq_rx:0')
    iq_tx = graph.get_tensor_by_name('iq_tx:0')
    ce_mean = graph.get_tensor_by_name('ce_mean:0')
    SNR = graph.get_tensor_by_name('SNR:0')
    return y, x, iq_receiver, outputs, total_loss, ber, berlin, conf_matrix, power_tx, noise_pwr, iq_rx, iq_tx, ce_mean, SNR


#########################################################
## Tx RX models
#########################################################


def dense_block_tx(inputs, FLAGS):
    '''
    This block is composed of fully connected layers, corresponding to channel encoder of Regular Comm System
    The size of coding block is (n_sc * nbits)
    :param inputs: tensor of size (batch, n_sym, n_sc, nbits)
    :param FLAGS: codein, codeout
    :return:
    '''
    if tf.__version__ == '1.4.0':
        activation_fn = tf.nn.relu
    else:
        activation_fn = tf.nn.leaky_relu
    codein = FLAGS.codein
    codeout = FLAGS.codeout
    _, n_sym, n_sc, nbits = inputs.shape
    n_sym, n_sc, nbits = int(n_sym), int(n_sc), int(nbits)
    n_sc_out = int(codeout * n_sc)//codein
    codeblockin = n_sc * nbits
    codeblockout = n_sc_out * nbits
    code0 = tf.reshape(inputs, [-1, codeblockin])
    # Traditional Coding
    code1 = tf.layers.dense(code0,
                          codeblockin,
                          kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                          bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                          activation=activation_fn # tf.nn.tanh
                          )
    # Emulate Turbo Coding (convolutional coding)
    out = tf.concat([code0, code1], axis=1)
    # out = tf.reshape(out, [-1, 2*codeblockin])
    out = tf.layers.dense(out,
                          codeblockout,
                          kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                          bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                          activation=tf.nn.tanh
                          )
    out = tf.reshape(out, [-1, n_sym, n_sc_out, nbits])
    return out



def conv_block_tx(inputs, FLAGS):
    if tf.__version__ == '1.4.0':
        activation_fn = tf.nn.relu
    else:
        activation_fn = tf.nn.leaky_relu
    m_iq = 2
    _, n_sym, n_sc, nbits = inputs.shape
    n_sym, n_sc, nbits = int(n_sym), int(n_sc), int(nbits)
    conv = inputs
    # Step 1: Constellation Mapping
    for i in range(4):
        conv = tf.layers.dense(conv,
                               2**nbits,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                               bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                               activation=tf.nn.tanh
                               )
    conv = tf.layers.conv2d(conv, m_iq, 1, strides=1, padding='same')
    conv = tf.nn.tanh(conv)
    conv = 2 * conv
    # Step 2 (optional): Waveform Shaping
    if FLAGS.conv:
        conv = tf.reshape(conv, [-1, n_sym, 1, n_sc, m_iq])
        # conv = layers_conv2d_complex(conv, n_sc, (1, 1), strides=1, padding='same')
        conv = layers_conv2d_complex(conv, n_sc, (n_sym, 1), strides=1, padding='same')
        conv = tf.reshape(conv, [-1, n_sym, n_sc, m_iq])

    return conv


def conv_block_rx(inputs, FLAGS):
    if tf.__version__ == '1.4.0':
        activation_fn = tf.nn.relu
    else:
        activation_fn = tf.nn.leaky_relu
    _, n_sym, n_sc, m_iq = inputs.shape
    K = 64
    nbits = FLAGS.nbits
    n_sym, n_sc, m_iq = int(n_sym), int(n_sc), int(m_iq)
    conv = inputs
    if FLAGS.conv:
        conv = tf.reshape(conv, [-1, n_sym, 1, n_sc, m_iq])
        conv = layers_conv2d_complex(conv, n_sc, (n_sym, 1), strides=1, padding='same')
        conv = tf.reshape(conv, [-1, n_sym, n_sc, m_iq])

    for i in range(4):
        conv = tf.layers.dense(conv,
                               2**nbits,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                               bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                               activation=activation_fn #tf.nn.tanh
                               )
    return conv


def dense_block_rx(inputs, FLAGS, outshape=None):
    if tf.__version__ == '1.4.0':
        activation_fn = tf.nn.relu
    else:
        activation_fn = tf.nn.leaky_relu
    _, frame_size, nbits, nllr = outshape
    frame_size, nbits, nllr = int(frame_size), int(nbits), int(nllr)
    _, in_sym, in_sc, nbits_exp = inputs.shape
    in_sym, in_sc, nbits_exp = int(in_sym), int(in_sc), int(nbits_exp)
    # assert(nbits == nbits1)
    encode_rx = int(in_sc * nbits)
    decode_rx = int(frame_size * nbits * nllr)

    out1 = tf.reshape(inputs, [-1, in_sym, in_sc * nbits_exp])
    out2 = tf.layers.dense(out1,
                          encode_rx,
                          kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                          bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                          activation=activation_fn# tf.nn.tanh
                          )
    # Emulate Turbo Coding (convolutional coding)
    out = tf.concat([out1, out2], axis=2)
    out = tf.layers.dense(out,
                          decode_rx,
                          kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                          bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                          activation=activation_fn
                          )
    out = tf.reshape(out, [-1, frame_size, nbits, nllr])
    out = tf.nn.softmax(out)
    return out

def equalizer_block(inputs, FLAGS, ofdmobj):
    '''
    Equalizer block for end 2 end communication
    :param inputs:
    :param FLAGS:
    :return:
    '''
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    n_filter = n_sym
    # inputs_complex = tf.complex(inputs[:,:,:,0],inputs[:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, n_sc, 1])
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    # if not FLAGS.cp:
    #     Kin = K
    #     chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, Kin, -1])
    # else:
    #     Kin = n_sc
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    ## Convert to Frequency Domain
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    # chest = tf.reshape(chest, [-1, n_sym, n_sc, 1, m_iq])
    chest = layers_conv2d_complex(chest, K, (1,K), strides=1, padding='valid')
    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])
    equalized_complex = tf.complex(chest[:, :, :, :, 0], chest[:, :, :, :, 1])

    layer_size = n_sym * P * m_iq
    for it in range(1):
        # chest0 = tf.reshape(chest, [-1, n_sym, n_sc, 1, m_iq])
        # chest0 = layers_conv2d_complex(chest0, n_filter, (1,n_sc), strides=1, padding='same')
        # chest0 = tf.reshape(chest0, [-1, n_filter*n_sym*n_sc*m_iq])
        inputs_complex = tf.reshape(equalized_complex, [-1, n_sym, K, 1])
        chest = tf.reshape(chest, [-1, n_sym*K*m_iq])
        chest = tf.layers.dense(chest,
                                 layer_size,
                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 #activation=tf.nn.tanh
                                 )
        chest0 = tf.layers.dense(chest,
                                 layer_size,
                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 activation=tf.nn.tanh
                                 )
        chest1 = tf.layers.dense(chest-chest0,
                                 layer_size,
                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 activation=tf.nn.tanh
                                 )
        chest2 = tf.layers.dense(chest0-chest1,
                                 layer_size,
                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 activation=tf.nn.tanh
                                 )
        chest3 = tf.layers.dense(chest1-chest2,
                                 layer_size,
                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 activation=tf.nn.tanh
                                 )
        chest4 = tf.layers.dense(chest2-chest3,
                                 layer_size,
                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 activation=tf.nn.tanh
                                 )
        chest5 = tf.layers.dense(chest3-chest4,
                                 layer_size,
                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                 activation=tf.nn.tanh
                                 )
        chest = tf.concat([chest0, chest1, chest2, chest3, chest4, chest5], axis=1)
        # chest = chest0

        chest = tf.layers.dense(chest,
                                n_sym * K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                #activation=tf.nn.tanh
                                )
        chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
        chest = layers_conv2d_complex(chest, 16, (n_sym,K), strides=1, padding='same')
        chest = tf.reshape(chest, [-1, 16*n_sym*K*m_iq])
        chest = tf.layers.dense(chest,
                                n_sym * K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activation=tf.nn.tanh
                                )
        chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
        chest = layers_conv2d_complex(chest, 1, (n_sym, K), strides=1, padding='same')
        chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

        chest_conj = tf.conj(chest)
        chest_square = tf.square(chest)
        equalized_complex = tf.multiply(inputs_complex,chest_conj)
        equalized_complex = tf.div(equalized_complex,chest_square)
        equalized_freq = tf.concat([tf.real(equalized_complex),tf.imag(equalized_complex)], axis=3)
        chest = equalized_freq

    equalized_complex = layers_conv2d_complex(equalized_complex, n_sc, (1,K), strides=1, padding='valid')
    equalized_complex = tf.transpose(equalized_complex, perm=[0, 1, 3, 2])
    equalized_complex = tf.reshape(equalized_complex, [-1,n_sym, n_sc, 1])
    equalized = tf.concat([tf.real(equalized_complex),tf.imag(equalized_complex)], axis=3)

    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])


    return equalized, snr_db



def equalizer_ofdm(inputs, FLAGS, ofdmobj):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    N_ESC = K - G - DC
    frame_size = ofdmobj.frame_size
    pilot_size = ofdmobj.pilot_size
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    ## Convert to Frequency Domain
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = layers_conv2d_complex(chest, K, (1, K), strides=1, padding='valid')
    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])

    # Frequency Domain Inputs IQ data
    inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)
    inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])
    # chest = tf.reshape(chest, [-1, n_sym, K, 1])
    # chest = tf.concat([tf.real(chest), tf.imag(chest)], axis=-1)

    pilot_size_iq = pilot_size * m_iq
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, n_sym * K * m_iq])

    pilot = tf.layers.dense(chest,
                            pilot_size_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = pilot

    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    # Optional Filtering Process

    n_out_blocks = 1
    for i in range(n_out_blocks):
        chest = tf.reshape(chest, [-1, n_sym * K * m_iq])
        # chest = tf.concat([chest, tf.nn.leaky_relu(chest)], axis=-1)
        chest = tf.layers.dense(chest,
                                n_sym * K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activation=tf.nn.tanh
                                )
        chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
        chest = layers_conv2d_complex(chest, 1, (n_sym, K), strides=(1, 1), padding='same')
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    chest_conj = tf.conj(chest)
    chest_abs = tf.abs(chest)
    chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
    chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])
    equalized_complex = tf.multiply(inputs_complex, chest_conj)
    equalized_freq = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=3)

    corr_cpx = tf.multiply(equalized_complex, tf.conj(equalized_complex))
    corr_cpx = layers_conv2d_complex(corr_cpx, K, (1,K), strides=1, padding='valid')
    corr_cpx = tf.transpose(corr_cpx,[0,1,3,2])
    corr_re = tf.concat([tf.real(corr_cpx), tf.imag(corr_cpx)], axis=-1)

    equalized_complex = layers_conv2d_complex(equalized_complex, K, (1, K), strides=1, padding='valid')
    equalized_complex = tf.transpose(equalized_complex, perm=[0, 1, 3, 2])
    # Directly Use FFT
    # equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K])
    # equalized_complex = tf.ifft(equalized_complex)
    equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K, 1])
    equalized = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=-1)

    # Try to mute the CP here
    # equalized = tf.reshape(equalized, [-1,n_sym, K*m_iq])
    # equalized_nl = tf.nn.leaky_relu(equalized)
    # equal_corr = tf.concat([equalized, corr_re, equalized_nl], axis=-1)
    # equalized = tf.reshape(equal_corr, [-1, n_sym, K*(3*m_iq)])
    equal_corr = tf.concat([equalized, corr_re], axis=-1)
    equalized = tf.reshape(equal_corr, [-1, n_sym, K * (2 * m_iq)])
    equalized = tf.layers.dense(equalized,
                                n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, n_sc, m_iq])

    # Calculate SNR dummy
    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    chest = tf.reshape(chest, [-1, n_sym, K])
    return equalized, snr_db, chest



def equalizer_nocconv(inputs, FLAGS, ofdmobj):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    N_ESC = K - G - DC
    frame_size = ofdmobj.frame_size
    pilot_size = ofdmobj.pilot_size
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    ## Convert to Frequency Domain
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])

    # Frequency Domain Inputs IQ data
    inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)
    inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])
    # chest = tf.reshape(chest, [-1, n_sym, K, 1])
    # chest = tf.concat([tf.real(chest), tf.imag(chest)], axis=-1)

    pilot_size_iq = pilot_size * m_iq
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, n_sym * K * m_iq])

    pilot = tf.layers.dense(chest,
                            pilot_size_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = pilot

    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )

    # cascading = [chest]
    n_out_blocks = 1
    for i in range(n_out_blocks):
        chest = tf.reshape(chest, [-1, n_sym * K * m_iq])
        # chest = tf.concat([chest, tf.nn.leaky_relu(chest)], axis=-1)
        chest = tf.layers.dense(chest,
                                n_sym * K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activation=tf.nn.tanh
                                )
        chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
        chest = layers_conv2d_complex(chest, 1, (n_sym, K), strides=(1, 1), padding='same')

    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    chest_conj = tf.conj(chest)
    chest_abs = tf.abs(chest)
    chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
    chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])
    equalized_complex = tf.multiply(inputs_complex, chest_conj)
    equalized_freq = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=3)

    equalized = tf.reshape(equalized_freq, [-1, n_sym, K * m_iq])
    equalized = tf.layers.dense(equalized,
                                K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, K, m_iq])
    # Directly Use FFT

    # Try to mute the CP here
    equal_corr = equalized
    equalized = tf.reshape(equal_corr, [-1, n_sym, K * m_iq])
    equalized = tf.layers.dense(equalized,
                                n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, n_sc, m_iq])

    # Calculate SNR dummy
    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    chest = tf.reshape(chest, [-1, n_sym, K])
    return equalized, snr_db, chest


def equalizer_noresdl(inputs, FLAGS, ofdmobj):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    N_ESC = K - G - DC
    frame_size = ofdmobj.frame_size
    pilot_size = ofdmobj.pilot_size
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    ## Convert to Frequency Domain
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = layers_conv2d_complex(chest, K, (1, K), strides=1, padding='valid')
    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])

    # Frequency Domain Inputs IQ data
    inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)
    inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])
    # chest = tf.reshape(chest, [-1, n_sym, K, 1])
    # chest = tf.concat([tf.real(chest), tf.imag(chest)], axis=-1)

    pilot_size_iq = pilot_size * m_iq
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, n_sym * K * m_iq])

    pilot = tf.layers.dense(chest,
                            pilot_size_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = pilot
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )

    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    chest_conj = tf.conj(chest)
    chest_abs = tf.abs(chest)
    chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
    chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])
    equalized_complex = tf.multiply(inputs_complex, chest_conj)
    equalized_freq = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=3)

    # equalized_complex = tf.signal.fft(equalized_complex)
    # equalized_complex = layers_conv2d_complex(equalized_complex, K, (1, K), strides=1, padding='valid')
    equalized_complex = tf.transpose(equalized_complex, perm=[0, 1, 3, 2])
    # Directly Use FFT
    equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K])
    equalized_complex = tf.ifft(equalized_complex)
    equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K, 1])
    equalized = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=-1)

    # Try to mute the CP here

    equal_corr = equalized
    equalized = tf.reshape(equal_corr, [-1, n_sym, K * m_iq])
    equalized = tf.layers.dense(equalized,
                                n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, n_sc, m_iq])

    # Calculate SNR
    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    chest = tf.reshape(chest, [-1, n_sym, K])
    return equalized, snr_db, chest



def equalizer_noresdl2(inputs, FLAGS, ofdmobj):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    N_ESC = K - G - DC
    frame_size = ofdmobj.frame_size
    pilot_size = ofdmobj.pilot_size
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    ## Convert to Frequency Domain
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = layers_conv2d_complex(chest, K, (1, K), strides=1, padding='valid')
    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])

    # Frequency Domain Inputs IQ data
    inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)
    inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])
    # chest = tf.reshape(chest, [-1, n_sym, K, 1])
    # chest = tf.concat([tf.real(chest), tf.imag(chest)], axis=-1)

    pilot_size_iq = pilot_size * m_iq
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, n_sym * K * m_iq])

    pilot = tf.layers.dense(chest,
                            pilot_size_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = pilot
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=tf.nn.tanh
                            )

    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    chest_conj = tf.conj(chest)
    chest_abs = tf.abs(chest)
    chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
    chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])
    equalized_complex = tf.multiply(inputs_complex, chest_conj)
    equalized_freq = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=3)

    # equalized_complex = tf.signal.fft(equalized_complex)
    # equalized_complex = layers_conv2d_complex(equalized_complex, K, (1, K), strides=1, padding='valid')
    equalized_complex = tf.transpose(equalized_complex, perm=[0, 1, 3, 2])
    # Directly Use FFT
    equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K])
    equalized_complex = tf.ifft(equalized_complex)
    equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K, 1])
    equalized = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=-1)

    # Try to mute the CP here

    equal_corr = equalized
    equalized = tf.reshape(equal_corr, [-1, n_sym, K * m_iq])
    equalized = tf.layers.dense(equalized,
                                n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, n_sc, m_iq])

    # Calculate SNR
    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    chest = tf.reshape(chest, [-1, n_sym, K])
    return equalized, snr_db, chest


def equalizer_noresdl4(inputs, FLAGS, ofdmobj):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    N_ESC = K - G - DC
    frame_size = ofdmobj.frame_size
    pilot_size = ofdmobj.pilot_size
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    ## Convert to Frequency Domain
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = layers_conv2d_complex(chest, K, (1, K), strides=1, padding='valid')
    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])

    # Frequency Domain Inputs IQ data
    inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)
    inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])
    # chest = tf.reshape(chest, [-1, n_sym, K, 1])
    # chest = tf.concat([tf.real(chest), tf.imag(chest)], axis=-1)

    pilot_size_iq = pilot_size * m_iq
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, n_sym * K * m_iq])

    pilot = tf.layers.dense(chest,
                            pilot_size_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = pilot
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=tf.nn.tanh
                            )
    #
    n_out_blocks = 2
    for i in range(n_out_blocks):
        chest = tf.reshape(chest, [-1, n_sym * K * m_iq])
        # chest = tf.concat([chest, tf.nn.leaky_relu(chest)], axis=-1)
        chest = tf.layers.dense(chest,
                                n_sym * K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activation=tf.nn.tanh
                                )
        chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    #     chest = layers_conv2d_complex(chest, 1, (n_sym, K), strides=(1, 1), padding='same')

    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    chest_conj = tf.conj(chest)
    chest_abs = tf.abs(chest)
    chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
    chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])
    equalized_complex = tf.multiply(inputs_complex, chest_conj)
    equalized_freq = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=3)

    # equalized_complex = tf.signal.fft(equalized_complex)
    # equalized_complex = layers_conv2d_complex(equalized_complex, K, (1, K), strides=1, padding='valid')
    equalized_complex = tf.transpose(equalized_complex, perm=[0, 1, 3, 2])
    # Directly Use FFT
    equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K])
    equalized_complex = tf.ifft(equalized_complex)
    equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K, 1])
    equalized = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=-1)

    # Try to mute the CP here

    equal_corr = equalized
    equalized = tf.reshape(equal_corr, [-1, n_sym, K * m_iq])
    equalized = tf.layers.dense(equalized,
                                n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, n_sc, m_iq])

    # Calculate SNR
    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    chest = tf.reshape(chest, [-1, n_sym, K])
    return equalized, snr_db, chest


def equalizer_dnnE(inputs, FLAGS, ofdmobj):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    N_ESC = K - G - DC
    frame_size = ofdmobj.frame_size
    pilot_size = ofdmobj.pilot_size
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    ## Convert to Frequency Domain
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])

    # Frequency Domain Inputs IQ data
    inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)
    inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])
    # chest = tf.reshape(chest, [-1, n_sym, K, 1])
    # chest = tf.concat([tf.real(chest), tf.imag(chest)], axis=-1)

    pilot_size_iq = pilot_size * m_iq
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, n_sym * K * m_iq])

    pilot = tf.layers.dense(chest,
                            pilot_size_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = pilot

    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=tf.nn.tanh
                            )
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=tf.nn.tanh
                            )

    # cascading = [chest]
    n_out_blocks = 2
    for i in range(n_out_blocks):
        chest = tf.reshape(chest, [-1, n_sym * K * m_iq])
        # chest = tf.concat([chest, tf.nn.leaky_relu(chest)], axis=-1)
        chest = tf.layers.dense(chest,
                                n_sym * K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activation=tf.nn.tanh
                                )
        chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
        # chest = layers_conv2d_complex(chest, 1, (n_sym, K), strides=(1, 1), padding='same')

    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    chest_conj = tf.conj(chest)
    chest_abs = tf.abs(chest)
    chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
    chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])
    equalized_complex = tf.multiply(inputs_complex, chest_conj)
    equalized_freq = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=3)

    # equalized_complex = layers_conv2d_complex(equalized_complex, K, (1, K), strides=1, padding='valid')
    # equalized_complex = tf.transpose(equalized_complex, perm=[0, 1, 3, 2])
    # equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K, 1])
    # equalized = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=-1)
    equalized = tf.reshape(equalized_freq, [-1, n_sym, K * m_iq])
    equalized = tf.layers.dense(equalized,
                                K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, K, m_iq])
    # Directly Use FFT

    # Try to mute the CP here
    equal_corr = equalized
    equalized = tf.reshape(equal_corr, [-1, n_sym, K * m_iq])
    equalized = tf.layers.dense(equalized,
                                n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, n_sc, m_iq])

    # Calculate SNR
    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    chest = tf.reshape(chest, [-1, n_sym, K])
    return equalized, snr_db, chest



def equalizer_separateIQ(inputs, FLAGS, ofdmobj):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    N_ESC = K - G - DC
    frame_size = ofdmobj.frame_size
    pilot_size = ofdmobj.pilot_size
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    ## Convert to Frequency Domain
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest =  layers_conv2d_vector(chest, K, (1, K), strides=1, padding='valid')
    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])

    # Frequency Domain Inputs IQ data
    inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)
    inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])
    # chest = tf.reshape(chest, [-1, n_sym, K, 1])
    # chest = tf.concat([tf.real(chest), tf.imag(chest)], axis=-1)

    pilot_size_iq = pilot_size * m_iq
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, n_sym * K * m_iq])

    pilot =  tf.layers.dense(chest,
                            pilot_size_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )
    chest = pilot

    chest =  tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=tf.nn.tanh
                            )
    chest =  tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=tf.nn.tanh
                            )

    # cascading = [chest]
    n_out_blocks = 1
    for i in range(n_out_blocks):
        chest = tf.reshape(chest, [-1, n_sym * K * m_iq])
        # chest = tf.concat([chest, tf.nn.leaky_relu(chest)], axis=-1)
        chest =  tf.layers.dense(chest,
                                n_sym * K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activation=tf.nn.tanh
                                )
        chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
        chest =  layers_conv2d_vector(chest, 1, (n_sym, K), strides=(1, 1), padding='same')

    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    chest_conj = tf.conj(chest)
    chest_abs = tf.abs(chest)
    chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
    chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])
    equalized_complex = tf.multiply(inputs_complex, chest_conj)
    equalized_freq = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=3)

    corr_cpx = tf.multiply(equalized_complex, tf.conj(equalized_complex))
    corr_cpx =  layers_conv2d_vector(corr_cpx, K, (1,K), strides=1, padding='valid')
    corr_cpx = tf.transpose(corr_cpx,[0,1,3,2])
    corr_re = tf.concat([tf.real(corr_cpx), tf.imag(corr_cpx)], axis=-1)

    equalized_complex =  layers_conv2d_vector(equalized_complex, K, (1, K), strides=1, padding='valid')
    equalized_complex = tf.transpose(equalized_complex, perm=[0, 1, 3, 2])
    # Directly Use FFT
    # equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K])
    # equalized_complex = tf.ifft(equalized_complex)
    equalized_complex = tf.reshape(equalized_complex, [-1, n_sym, K, 1])
    equalized = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=-1)

    # Try to mute the CP here
    # equalized = tf.reshape(equalized, [-1,n_sym, K*m_iq])
    # equalized_nl = tf.nn.leaky_relu(equalized)
    # equal_corr = tf.concat([equalized, corr_re, equalized_nl], axis=-1)
    # equalized = tf.reshape(equal_corr, [-1, n_sym, K*(3*m_iq)])
    equal_corr = tf.concat([equalized, corr_re], axis=-1)
    equalized = tf.reshape(equal_corr, [-1, n_sym, K * (2 * m_iq)])
    equalized = tf.layers.dense(equalized,
                                n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    equalized = tf.reshape(equalized, [-1, n_sym, n_sc, m_iq])

    # Calculate SNR
    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    chest = tf.reshape(chest, [-1, n_sym, K])
    return equalized, snr_db, chest



def ofdm_dense_rx(inputs, FLAGS, ofdmobj, outshape=None):
    if tf.__version__ == '1.4.0':
        activation_fn = tf.nn.relu
    else:
        activation_fn = tf.nn.leaky_relu
    _, n_sym, n_sc, m_iq = inputs.shape
    n_filters = FLAGS.nfilter
    n_esc = ofdmobj.K - ofdmobj.G - ofdmobj.DC
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    n_sym, n_sc, m_iq = int(n_sym), int(n_sc), int(m_iq)
    out = inputs
    # out = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # Remove CP
    if not FLAGS.cp:
        K = ofdmobj.K
        out = tf.slice(out, [0, 0, CP, 0], [-1, -1, K, -1])
    else:
        K = n_sc
    n_sc = n_filters
    _, data_ofdm, nbits, nllr = outshape
    data_ofdm, nbits, nllr = int(data_ofdm), int(nbits), int(nllr)

    # layer 1: convolutional FFT like
    with tf.variable_scope('fft_like') as scope:
        # Option 1: C-Conv Layer
        conv = tf.reshape(out, [-1, n_sym, 1, K, m_iq])
        out = layers_conv2d_complex(conv, n_sc, (1, K), strides=1, padding='same')
        # Option 2: Dense Layer
        # out = tf.reshape(out, [-1, n_sym * K * m_iq])
        # out = tf.layers.dense(out,
        #                       n_sym * n_sc * m_iq,
        #                       kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
        #                       bias_regularizer=tf.keras.regularizers.l2(l=0.01),
        #                       )
        # Option 3: FFT
        # out = tf.complex(out[:, :, :, 0], out[:, :, :, 1])
        # out = tf.fft(out)
        # out = tf.reshape(out, [-1, n_sym, n_sc, 1])
        # out = tf.concat([tf.real(out), tf.imag(out)], axis=-1)
        out = tf.reshape(out, [-1, n_sym, n_sc, m_iq])
        out0 = out
        tf.identity(out0, 'fft_out')

    # Layer 2: Data IQ extraction
    with tf.variable_scope('demodulation') as scope:
        out = tf.reshape(out, [-1, n_sym * n_sc * m_iq])
        out = tf.layers.dense(out,
                              data_ofdm * m_iq,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              # activation=tf.nn.tanh,
                              )
        out = tf.reshape(out, [-1, 1, data_ofdm, m_iq])
        out_iq = out

        out = tf.layers.conv2d(out, 2**nbits, 1, strides=1, padding='same')
        # out = tf.layers.conv2d(out, 2**nbits, 1, strides=1, padding='same')
        out = tf.nn.leaky_relu(out)

        out = tf.concat([out, out_iq], axis=-1)
        out = tf.layers.dense(out,
                              nbits * nllr,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              activation=tf.nn.leaky_relu,
                              )

    out = tf.reshape(out, [-1, data_ofdm, nbits, nllr])
    out = tf.nn.softmax(out)
    return out


def equalizer_freq(inputs, FLAGS, ofdmobj):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            )
    ## Convert to Frequency Domain
    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])

    # Frequency Domain Inputs IQ data
    inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)
    inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])
    # chest = tf.reshape(chest, [-1, n_sym, K, 1])
    # chest = tf.concat([tf.real(chest), tf.imag(chest)], axis=-1)

    pilot_size = n_sym * P * m_iq
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, n_sym * K * m_iq])

    pilot = tf.layers.dense(chest,
                            pilot_size,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    chest = pilot

    chest0 = tf.layers.dense(chest,
                            pilot_size,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    cascading = [pilot, chest0]
    for i in range(4):
        chest1 = tf.layers.dense(chest-chest0,
                                pilot_size,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                #activation=tf.nn.tanh
                                )
        chest1 = tf.reshape(chest1, [-1, n_sym, P, 1, m_iq])
        chest1 = layers_conv2d_complex(chest1, 1, (1, P), strides=1, padding='same')
        chest1 = tf.reshape(chest1, [-1, n_sym * P * m_iq])
        cascading.append(chest1)
        chest = chest0
        chest0 = chest1


    # chest = tf.reshape(chest, [-1, P*m_iq])
    chest = tf.concat(cascading, axis=-1)
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    chest = tf.layers.dense(chest,
                            n_sym * K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            #activation=tf.nn.tanh
                            )
    # Optional Filtering Process
    # chest = tf.reshape(chest, [-1, n_sym*K, 1, m_iq])
    # for i in range(1):
    #     chest = layers_conv1d_complex(chest, 1, int(n_sym*K), strides=1, padding='same')
    # chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])

    cascading = []
    n_out_blocks = 1
    for i in range(n_out_blocks):
        chest = tf.reshape(chest, [-1, n_sym * K * m_iq])
        chest = tf.layers.dense(chest,
                                n_sym * K * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activation=tf.nn.tanh
                                )
        chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
        chest = layers_conv2d_complex(chest, 1, (n_sym, K), strides=1, padding='same')
        # cascading.append(chest)

    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    # Equalization
    chest_conj = tf.conj(chest)
    chest_abs = tf.abs(chest)
    chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
    chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])

    equalized_complex = tf.multiply(inputs_complex, chest_conj)
    equalized_freq = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=3)
    chest = equalized_freq

    # Calculate SNR
    equalized_pilots_list = []
    for sc in pilotCarriers:
        equalized_pilots_list.append(equalized_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1,P,m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:,:,0],pilot_CPX[:,:,1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.log(snr_est) / tf.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    return chest, snr_db


def ofdm_equalized_rx(inputs, FLAGS, ofdmobj, outshape=None):
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    N_ESC = K - G - DC
    frame_size = ofdmobj.frame_size
    pilot_size = ofdmobj.pilot_size
    n_filters = FLAGS.nfilter

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    y_cp = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # chest = inputs
    if not FLAGS.cp:
        K = ofdmobj.K
        y_cp = tf.slice(y_cp, [0, 0, CP, 0], [-1, -1, K, -1])
    else:
        K = n_sc
    n_sc = n_filters
    _, data_ofdm, nbits, nllr = outshape
    data_ofdm, nbits, nllr = int(data_ofdm), int(nbits), int(nllr)

    y_cp = tf.reshape(y_cp, [-1, n_sym, K * m_iq])
    y_cp = tf.layers.dense(y_cp,
                            n_sc * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=tf.nn.tanh
                            )

    # layer 1: convolutional FFT like
    with tf.variable_scope('fft_like') as scope:
        # Option 1: C-Conv Layer
        conv = tf.reshape(y_cp, [-1, n_sym, 1, n_sc, m_iq])
        chest = layers_conv2d_complex(conv, n_sc, (1, n_sc), strides=1, padding='same')
        # chest = tf.reshape(chest, [-1, n_sym, n_sc, m_iq])
        tf.identity(chest, 'fft_out')

    chest = tf.transpose(chest, perm=[0, 1, 3, 2, 4])

    with tf.variable_scope('channel_estimation') as scope:
        # Frequency Domain Inputs IQ data
        inputs_complex = tf.complex(chest[:,:,:,:,0],chest[:,:,:,:,1])
        # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K, 1])

        pilot_size_iq = pilot_size * m_iq
        ## Pilot Extraction & LS Channel Estimation
        chest = tf.reshape(chest, [-1, n_sym * n_sc * m_iq])
        pilot = tf.layers.dense(chest,
                                pilot_size_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                # activation=tf.nn.tanh
                                )
        chest = tf.layers.dense(pilot,
                                n_sym * n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                #activation=tf.nn.tanh
                                )
        chest = tf.layers.dense(chest,
                                n_sym * n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                #activation=tf.nn.tanh
                                )
        n_out_blocks = 1
        for i in range(n_out_blocks):
            # chest = tf.reshape(chest, [-1, n_sym * n_sc * m_iq])
            # # chest = tf.concat([chest, tf.nn.leaky_relu(chest)], axis=-1)
            # chest = tf.layers.dense(chest,
            #                         n_sym * n_sc * m_iq,
            #                         kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
            #                         bias_regularizer=tf.keras.regularizers.l2(l=0.01),
            #                         activation=tf.nn.tanh
            #                         )
            chest = tf.reshape(chest, [-1, n_sym, n_sc, 1, m_iq])
            chest = layers_conv2d_complex(chest, 1, (n_sym, n_sc), strides=(1, 1), padding='same')
        chest = tf.reshape(chest, [-1, n_sym, n_sc, 1, m_iq])
        chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

        chest_conj = tf.conj(chest)
        chest_abs = tf.abs(chest)
        chest_conj = tf.concat([tf.div(tf.real(chest_conj), chest_abs), tf.div(tf.imag(chest_conj), chest_abs)], axis=-1)
        chest_conj = tf.complex(chest_conj[:,:,:,0:1], chest_conj[:,:,:,1:])
        equalized_complex = tf.multiply(inputs_complex, chest_conj)

    equalized = tf.concat([tf.real(equalized_complex), tf.imag(equalized_complex)], axis=-1)

    # Layer 2: Data IQ extraction
    with tf.variable_scope('demodulation') as scope:
        out = tf.reshape(equalized, [-1, n_sym * n_sc * m_iq])
        out = tf.layers.dense(out,
                              data_ofdm * m_iq,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              # activation=tf.nn.tanh,
                              )
        out = tf.reshape(out, [-1, data_ofdm, m_iq])
        out_iq = out

        out = tf.nn.leaky_relu(out)

        out = tf.concat([out, out_iq], axis=-1)
        out = tf.layers.dense(out,
                              nbits * nllr,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              activation=tf.nn.leaky_relu,
                              )

    out = tf.reshape(out, [-1, data_ofdm, nbits, nllr])
    out = tf.nn.softmax(out)
    return out



def ofdm_DNN_rx(inputs, FLAGS, ofdmobj, outshape=None):
    if tf.__version__ == '1.4.0':
        activation_fn = tf.nn.relu
    else:
        activation_fn = tf.nn.leaky_relu
    _, n_sym, n_sc, m_iq = inputs.shape
    n_filters = FLAGS.nfilter
    n_esc = ofdmobj.K - ofdmobj.G - ofdmobj.DC
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    n_sym, n_sc, m_iq = int(n_sym), int(n_sc), int(m_iq)
    out = inputs
    # out = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    # Remove CP
    if not FLAGS.cp:
        K = ofdmobj.K
        out = tf.slice(out, [0, 0, CP, 0], [-1, -1, K, -1])
    else:
        K = n_sc
    n_sc = n_filters
    _, data_ofdm, nbits, nllr = outshape
    data_ofdm, nbits, nllr = int(data_ofdm), int(nbits), int(nllr)

    # layer 1: convolutional FFT like
    with tf.variable_scope('fft_like') as scope:
        # Option 1: C-Conv Layer
        # conv = tf.reshape(out, [-1, n_sym, 1, K, m_iq])
        # out = layers_conv2d_complex(conv, n_sc, (1, K), strides=1, padding='same')
        # Option 2: Dense Layer
        out = tf.reshape(out, [-1, n_sym, K * m_iq])
        out = tf.layers.dense(out,
                              n_sc * m_iq,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              activation=tf.nn.leaky_relu
                              )
        out = tf.layers.dense(out,
                              250,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              activation=tf.nn.leaky_relu
                              )
        out = tf.layers.dense(out,
                              125,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              activation=tf.nn.leaky_relu
                              )
        out = tf.layers.dense(out,
                              data_ofdm * m_iq,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              activation=tf.nn.leaky_relu
                              )
        # Option 3: FFT
        # out = tf.complex(out[:, :, :, 0], out[:, :, :, 1])
        # out = tf.fft(out)
        # out = tf.reshape(out, [-1, n_sym, n_sc, 1])
        # out = tf.concat([tf.real(out), tf.imag(out)], axis=-1)
        # out = tf.reshape(out, [-1, n_sym, n_sc, m_iq])
        # out0 = out
        # tf.identity(out0, 'fft_out')

    # Layer 2: Data IQ extraction
    with tf.variable_scope('demodulation') as scope:
        # out = tf.reshape(out, [-1, n_sym * n_sc * m_iq])
        out = tf.layers.dense(out,
                              data_ofdm * m_iq,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              activation=tf.nn.leaky_relu,
                              )
        out = tf.reshape(out, [-1, n_sym, data_ofdm, m_iq])
        # out_iq = out


        # out = tf.concat([out, out_iq], axis=-1)
        out = tf.layers.dense(out,
                              nbits * nllr,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                              activation=tf.nn.leaky_relu,
                              )

    out = tf.reshape(out, [-1, data_ofdm, nbits, nllr])
    out = tf.nn.softmax(out)
    return out



def equalizer_dnn(inputs, FLAGS, ofdmobj):
    if tf.__version__ == '1.4.0':
        activation_fn = tf.nn.relu
    else:
        activation_fn = tf.nn.leaky_relu
    K = ofdmobj.K # 64  # number of OFDM subcarriers
    CP = ofdmobj.CP  # length of the cyclic prefix: 25% of the block
    P = ofdmobj.P # 8  # number of pilot carriers per OFDM symbol
    G = ofdmobj.G # 8  # number of guard subcarriers per OFDM symbol
    DC = ofdmobj.DC
    n_filters = FLAGS.nfilter

    pilotCarriers = ofdmobj.pilotCarriers.astype(np.int32)

    _, n_sym, n_sc, m_iq = inputs.get_shape()
    n_sym, n_sc, m_iq = int(n_sym), int(n_sc), int(m_iq)
    # chest = tf.contrib.layers.layer_norm(inputs, center=False, scale=False, begin_norm_axis=1)
    layer_norm = tf.keras.layers.LayerNormalization(axis=1, center=False, scale=False)
    chest = layer_norm(inputs)
    # chest = inputs
    if not FLAGS.cp:
        chest = tf.slice(chest, [0, 0, CP, 0], [-1, -1, K, -1])
        chest = tf.reshape(chest, [-1, n_sym, K * m_iq])
    else:
        chest = tf.reshape(chest, [-1, n_sym, n_sc * m_iq])
    chest = tf.layers.dense(chest,
                            K * m_iq,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=activation_fn
                            )
    # Frequency Domain Inputs IQ data
    # inputs_complex = tf.reshape(inputs_complex, [-1, n_sym, K])
    # chest = tf.fft(inputs_complex)

    pilot_size = n_sym * P * m_iq
    frame_size = n_sym * K * m_iq
    inputs_complex = tf.reshape(chest, [-1, n_sym * K * m_iq])
    ## Pilot Extraction & LS Channel Estimation
    chest = tf.reshape(chest, [-1, frame_size])

    chest = tf.layers.dense(chest,
                            pilot_size,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=activation_fn
                            )

    chest = tf.layers.dense(chest,
                            pilot_size * 2 - 8,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=activation_fn
                            )
    chest = tf.layers.dense(chest,
                            frame_size,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=activation_fn
                            )

    cascading = [inputs_complex, chest]
    iq_freq = tf.concat(cascading, axis=-1)
    iq_freq = tf.layers.dense(iq_freq,
                            frame_size * 2 - 30,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            # activation=activation_fn
                            )

    iq_freq = tf.layers.dense(iq_freq,
                            frame_size,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activation=activation_fn
                            )

    chest = tf.reshape(chest, [-1, n_sym, K, 1, m_iq])
    chest = tf.complex(chest[:,:,:,:,0], chest[:,:,:,:,1])

    iq_time = tf.reshape(iq_freq, [-1, n_sym, K*m_iq])
    iq_time = tf.layers.dense(iq_time,
                                n_sc * m_iq,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                )
    iq_time = tf.reshape(iq_time, [-1, n_sym, n_sc, m_iq])

    # Calculate SNR
    equalized_pilots_list = []
    iq_freq = tf.reshape(iq_freq, [-1, n_sym, K, m_iq])
    for sc in pilotCarriers:
        equalized_pilots_list.append(iq_freq[:, :, sc:sc+1, :])
    equalized_pilots = tf.concat(equalized_pilots_list, axis=2)
    pilot_CPX = tf.reshape(equalized_pilots,[-1, P, m_iq])
    pilot_CPX = tf.complex(pilot_CPX[:, :, 0], pilot_CPX[:, :, 1])
    freq_cpx = tf.reshape(pilot_CPX, [-1, n_sym * P])
    signal_pwr, noise_pwr = tf.nn.moments(tf.square(tf.abs(freq_cpx)), axes=[1], keep_dims=True)
    snr_est = tf.clip_by_value(signal_pwr/noise_pwr, 0.001, 10000.0)
    snr_db = tf.math.log(snr_est) / tf.math.log(10.)
    snr_db = tf.reshape(snr_db, [-1, 1])

    chest = tf.reshape(chest, [-1, n_sym, K])
    return iq_time, snr_db, chest
