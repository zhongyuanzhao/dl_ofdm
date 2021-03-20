#!/usr/bin/env python
######################################################################################
# Library for complex operations and layers in tensorflow
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


def complex_clip(inputs, peak=1.):
    shapes = inputs.get_shape()
    assert(shapes[-1] == 2)
    clipped_inputs = tf.clip_by_norm(inputs, peak, axes=[-1])
    # power_tx = tf.reduce_mean(tf.square(tf.norm(clipped_inputs, axis=-1)))
    power_tx = tf.reduce_mean(tf.square(clipped_inputs[:,:,:,0])+tf.square(clipped_inputs[:,:,:,1]))
    return clipped_inputs, power_tx


def nn_conv1d_complex(inputs, filter):
    '''
    Perform complex 1D convolution operation
    :param inputs: 4D tensor, batch + length + channel + IQ
    :param filter: 4D tensor, filtersize + channel + 1 + IQ
    :return: 3D tensor, batch + length + IQ
    '''
    inputs_shape = inputs.get_shape()
    filter_shape = filter.get_shape()
    assert(inputs_shape[-1] == 2)
    assert(filter_shape[-1] == 2)
    assert(inputs_shape[2] == filter_shape[1])
    assert(filter_shape[2] == 1)

    output_re = tf.nn.conv1d(inputs[:,:,:,0], filter[:,:,:,0], stride=1, padding="SAME") - tf.nn.conv1d(inputs[:,:,:,1], filter[:,:,:,1], stride=1, padding="SAME")
    output_im = tf.nn.conv1d(inputs[:,:,:,0], filter[:,:,:,1], stride=1, padding="SAME") + tf.nn.conv1d(inputs[:,:,:,1], filter[:,:,:,0], stride=1, padding="SAME")
    # put real and image parts together without making complex number
    output = tf.concat([output_re, output_im], axis=2)
    return output


def layers_conv1d_complex(inputs, filters, kernal, strides=1, padding='valid'):
    '''
    Implement 1-D complex convolution layer based on real 2-D convolution layer
    :param inputs: 4-D real or 3-D complex tensor, [batch, size, channel, IQ(2)]
    :param filters: number of filters
    :param kernal: size of kernal
    :param strides:
    :param padding:
    :return: 4-D real or 3-D complex tensor, [batch, size, filters, IQ(2)]
    '''
    shapes = inputs.get_shape()
    rank = len(shapes)
    dtype = inputs.dtype
    assert(type(kernal) == int)
    if rank == 3 and dtype == tf.complex64:
        inputs_re = tf.real(inputs)
        inputs_im = tf.imag(inputs)
        inputs_re = tf.reshape(inputs_re, [-1, shapes[1], shapes[2], 1])
        inputs_im = tf.reshape(inputs_im, [-1, shapes[1], shapes[2], 1])
        inputs = tf.concat([inputs_re, inputs_im], axis=3)
        complex_flag = True
    elif rank == 4 and shapes[-1] == 2:
        complex_flag = False
        pass
    else:
        raise NameError('Check input tensor dtypes or shape')

    conv = tf.transpose(inputs, perm=[0, 1, 3, 2])

    conv = tf.layers.conv2d(conv, filters*2, (kernal,1), strides=strides, padding=padding)
    conv = tf.reshape(conv, [-1, shapes[1], shapes[3]*2, filters])
    shapes_conv = conv.get_shape().as_list()
    assert(shapes[3] == 2)
    conv_re = conv[:,:,0,:] - conv[:,:,3,:]
    conv_im = conv[:,:,1,:] - conv[:,:,2,:]
    conv_re = tf.reshape(conv_re, [-1, shapes_conv[1], 1, filters])
    conv_im = tf.reshape(conv_im, [-1, shapes_conv[1], 1, filters])
    output = tf.concat([conv_re,conv_im], axis=2)
    output = tf.transpose(output, perm=[0, 1, 3, 2])
    if complex_flag:
        output = tf.complex(output[:,:,:,0],output[:,:,:,1])
    return output


def layers_conv1d_transpose_complex(inputs, filters, kernal, strides=1, padding='valid'):
    '''
    Implement 1-D complex convolution layer based on real 2-D convolution layer
    :param inputs: 4-D real or 3-D complex tensor, [batch, size, channel, IQ(2)]
    :param filters: number of filters
    :param kernal: size of kernal
    :param strides:
    :param padding:
    :return: 4-D real or 3-D complex tensor, [batch, size, filters, IQ(2)]
    '''
    shapes = inputs.get_shape()
    rank = len(shapes)
    dtype = inputs.dtype
    assert(type(kernal) == int)
    if rank == 3 and dtype == tf.complex64:
        inputs_re = tf.real(inputs)
        inputs_im = tf.imag(inputs)
        inputs_re = tf.reshape(inputs_re, [-1, shapes[1], shapes[2], 1])
        inputs_im = tf.reshape(inputs_im, [-1, shapes[1], shapes[2], 1])
        inputs = tf.concat([inputs_re, inputs_im], axis=3)
        complex_flag = True
    elif rank == 4 and shapes[-1] == 2:
        complex_flag = False
        pass
    else:
        raise NameError('Check input tensor dtypes or shape')

    conv = tf.transpose(inputs, perm=[0, 1, 3, 2])

    conv = tf.layers.conv2d_transpose(conv, filters*2, (kernal,1), strides=strides, padding=padding)
    conv = tf.reshape(conv, [-1, shapes[1], shapes[3]*2, filters])
    shapes_conv = conv.get_shape().as_list()
    assert(shapes[3] == 2)
    conv_re = conv[:,:,0,:] - conv[:,:,3,:]
    conv_im = conv[:,:,1,:] - conv[:,:,2,:]
    conv_re = tf.reshape(conv_re, [-1, shapes_conv[1], 1, filters])
    conv_im = tf.reshape(conv_im, [-1, shapes_conv[1], 1, filters])
    output = tf.concat([conv_re,conv_im], axis=2)
    output = tf.transpose(output, perm=[0, 1, 3, 2])
    if complex_flag:
        output = tf.complex(output[:,:,:,0],output[:,:,:,1])
    return output



def layers_conv2d_complex(inputs, filters, kernal, strides=1, padding='valid'):
    '''
    Implement 2-D complex convolution layer based on real 3-D convolution layer
    :param inputs: 5-D real or 4-D complex tensor, [batch, length, width, channel, IQ(2)]
    :param filters: number of filters
    :param kernal: size of kernal
    :param strides:
    :param padding:
    :return: 5-D real or 4-D complex tensor, [batch, length, width, filters, IQ(2)]
    '''
    shapes = inputs.get_shape().as_list()
    rank = len(shapes)
    dtype = inputs.dtype
    assert (len(kernal) < 3)
    if rank == 4 and dtype == tf.complex64:
        inputs_re = tf.real(inputs)
        inputs_im = tf.imag(inputs)
        inputs_re = tf.reshape(inputs_re, [-1, shapes[1], shapes[2], shapes[3], 1])
        inputs_im = tf.reshape(inputs_im, [-1, shapes[1], shapes[2], shapes[3], 1])
        inputs = tf.concat([inputs_re, inputs_im], axis=4)
        complex_flag = True
        shapes.append(2)
    elif rank == 5 and shapes[-1] == 2:
        complex_flag = False
        pass
    else:
        raise TypeError('Check input tensor dtypes or shape')

    conv = tf.transpose(inputs, perm=[0, 1, 2, 4, 3])
    if type(kernal) is int:
        kernal_size = (kernal,kernal,1)
    elif type(kernal) is tuple and len(kernal) == 2:
        kernal_size = kernal + (1,)
    else:
        raise NameError('Unacceptable Kernal Size')

    if type(strides) is int:
        strides = (strides,strides,1)
    elif type(strides) is tuple and len(strides) == 2:
        strides = strides + (1,)
    else:
        raise NameError('Unacceptable Kernal Size')

    conv = tf.layers.conv3d(conv, filters * 2, kernal_size, strides=strides, padding=padding)
    shapes_conv = conv.get_shape().as_list()
    conv = tf.reshape(conv, [-1, shapes_conv[1], shapes_conv[2], shapes_conv[3] * 2, filters])
    assert (shapes[4] == 2)
    conv_re = conv[:, :, :, 0, :] - conv[:, :, :, 3, :]
    conv_im = conv[:, :, :, 1, :] - conv[:, :, :, 2, :]
    conv_re = tf.reshape(conv_re, [-1, shapes_conv[1], shapes_conv[2], 1, filters])
    conv_im = tf.reshape(conv_im, [-1, shapes_conv[1], shapes_conv[2], 1, filters])
    output = tf.concat([conv_re, conv_im], axis=3)
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
    if complex_flag:
        output = tf.complex(output[:, :, :, :, 0], output[:, :, :, :, 1])

    return output


def layers_conv2d_vector(inputs, filters, kernal, strides=1, padding='valid'):
    '''
    Approximate implementtation of 2D complex convolution layer based on real 3-D convolution layer
    :param inputs: 5-D real or 4-D complex tensor, [batch, length, width, channel, IQ(2)]
    :param filters: number of filters
    :param kernal: size of kernal
    :param strides:
    :param padding:
    :return: 5-D real or 4-D complex tensor, [batch, length, width, filters, IQ(2)]
    '''
    shapes = inputs.get_shape().as_list()
    rank = len(shapes)
    dtype = inputs.dtype
    assert (len(kernal) < 3)
    if rank == 4 and dtype == tf.complex64:
        inputs_re = tf.real(inputs)
        inputs_im = tf.imag(inputs)
        inputs_re = tf.reshape(inputs_re, [-1, shapes[1], shapes[2], shapes[3], 1])
        inputs_im = tf.reshape(inputs_im, [-1, shapes[1], shapes[2], shapes[3], 1])
        inputs = tf.concat([inputs_re, inputs_im], axis=4)
        complex_flag = True
        shapes.append(2)
    elif rank == 5 and shapes[-1] == 2:
        complex_flag = False
        pass
    else:
        raise TypeError('Check input tensor dtypes or shape')

    conv = tf.transpose(inputs, perm=[0, 1, 2, 4, 3])
    if type(kernal) is int:
        kernal_size = (kernal,kernal,1)
    elif type(kernal) is tuple and len(kernal) == 2:
        kernal_size = kernal + (2,)
    else:
        raise NameError('Unacceptable Kernal Size')

    if type(strides) is int:
        strides = (strides,strides,1)
    elif type(strides) is tuple and len(strides) == 2:
        strides = strides + (1,)
    else:
        raise NameError('Unacceptable Kernal Size')

    conv = tf.layers.conv3d(conv, filters*2, kernal_size, strides=strides, padding=padding)
    shapes_conv = conv.get_shape().as_list()
    conv = tf.reshape(conv, [-1, shapes_conv[1], shapes_conv[2], shapes_conv[3]*2, filters])
    assert (shapes[4] == 2)
    conv_re = conv[:, :, :, 0, :]
    conv_im = conv[:, :, :, 1, :]
    conv_re = tf.reshape(conv_re, [-1, shapes_conv[1], shapes_conv[2], 1, filters])
    conv_im = tf.reshape(conv_im, [-1, shapes_conv[1], shapes_conv[2], 1, filters])
    output = tf.concat([conv_re, conv_im], axis=3)
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
    if complex_flag:
        output = tf.complex(output[:, :, :, :, 0], output[:, :, :, :, 1])

    return output


def layers_conv2d_streams(inputs, filters, kernal, strides=1, padding='valid'):
    '''
    Oversimplified implementation of 2-D complex convolution layer by two independent real 2-D convolution layers
    :param inputs: 5-D real or 4-D complex tensor, [batch, length, width, channel, IQ(2)]
    :param filters: number of filters
    :param kernal: size of kernal
    :param strides:
    :param padding:
    :return: 5-D real or 4-D complex tensor, [batch, length, width, filters, IQ(2)]
    '''
    shapes = inputs.get_shape().as_list()
    rank = len(shapes)
    dtype = inputs.dtype
    assert (len(kernal) < 3)
    if rank == 4 and dtype == tf.complex64:
        inputs_re = tf.real(inputs)
        inputs_im = tf.imag(inputs)
        inputs_re = tf.reshape(inputs_re, [-1, shapes[1], shapes[2], shapes[3]])
        inputs_im = tf.reshape(inputs_im, [-1, shapes[1], shapes[2], shapes[3]])
        # inputs = tf.concat([inputs_re, inputs_im], axis=3)
        complex_flag = True
        shapes.append(2)
    elif rank == 5 and shapes[-1] == 2:
        inputs_re = inputs[:, :, :, :, 0]
        inputs_im = inputs[:, :, :, :, 1]
        complex_flag = False
        pass
    else:
        raise TypeError('Check input tensor dtypes or shape')

    # conv = tf.transpose(inputs, perm=[0, 1, 2, 4, 3])
    if type(kernal) is int:
        kernal_size = (kernal, kernal)
    elif type(kernal) is tuple and len(kernal) == 2:
        kernal_size = kernal
    else:
        raise NameError('Unacceptable Kernal Size')

    if type(strides) is int:
        strides = (strides, strides)
    elif type(strides) is tuple and len(strides) == 2:
        strides = strides
    else:
        raise NameError('Unacceptable Kernal Size')

    # conv = tf.layers.conv3d(conv, filters * 2, kernal_size, strides=strides, padding=padding)
    conv_re = tf.layers.conv2d(inputs_re, filters, kernal_size, strides=strides, padding=padding)
    conv_im = tf.layers.conv2d(inputs_im, filters, kernal_size, strides=strides, padding=padding)
    shapes_conv = conv_re.get_shape().as_list()
    # conv = tf.reshape(conv, [-1, shapes_conv[1], shapes_conv[2], shapes_conv[3] * 2, filters])
    assert (shapes[4] == 2)
    # conv_re = conv[:, :, :, 0, :] - conv[:, :, :, 3, :]
    # conv_im = conv[:, :, :, 1, :] - conv[:, :, :, 2, :]
    conv_re = tf.reshape(conv_re, [-1, shapes_conv[1], shapes_conv[2], 1, filters])
    conv_im = tf.reshape(conv_im, [-1, shapes_conv[1], shapes_conv[2], 1, filters])
    output = tf.concat([conv_re, conv_im], axis=3)
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
    if complex_flag:
        output = tf.complex(output[:, :, :, :, 0], output[:, :, :, :, 1])

    return output


def layers_dense_streams(inputs, outdim, kernel_regularizer=None, bias_regularizer=None, activation=None):
    '''
    Oversimplified implementation of complex dense layer by two independent dense layers
    '''
    shapes = inputs.get_shape().as_list()
    outshapes = shapes[:]
    outdim = int(outdim)
    outshapes[-1] = outdim
    outshapes = [-1 if x is None else x for x in outshapes]
    rank = len(shapes)
    if rank >= 2:
        inputs = tf.reshape(inputs, [-1, shapes[-1]])
        assert(shapes[-1] % 2 == 0)
        assert(outdim % 2 == 0)
        halfindim = int(shapes[1]/2)
        halfoutdim = int(outdim/2)
        inputs_re = inputs[:, 0:halfindim]
        inputs_im = inputs[:, halfindim:]
        pass
    else:
        raise TypeError('Check input tensor dtypes or shape')
    outputs_re = tf.layers.dense(inputs_re,
                            halfoutdim,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            activation=activation
                            )
    outputs_im = tf.layers.dense(inputs_im,
                            halfoutdim,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            activation=activation
                            )
    outputs = tf.concat([outputs_re, outputs_im], axis=1)
    outputs = tf.reshape(outputs, outshapes)
    return outputs


def layers_conv2d_transpose_complex(inputs, filters, kernal, strides=1, padding='valid'):
    '''
    Implement 2-D complex convolution layer based on real 3-D convolution layer
    :param inputs: 5-D real or 4-D complex tensor, [batch, length, width, channel, IQ(2)]
    :param filters: number of filters
    :param kernal: size of kernal
    :param strides:
    :param padding:
    :return: 5-D real or 4-D complex tensor, [batch, length, width, filters, IQ(2)]
    '''
    shapes = inputs.get_shape().as_list()
    rank = len(shapes)
    dtype = inputs.dtype
    assert (len(kernal) < 3)
    if rank == 4 and dtype == tf.complex64:
        inputs_re = tf.real(inputs)
        inputs_im = tf.imag(inputs)
        inputs_re = tf.reshape(inputs_re, [-1, shapes[1], shapes[2], shapes[3], 1])
        inputs_im = tf.reshape(inputs_im, [-1, shapes[1], shapes[2], shapes[3], 1])
        inputs = tf.concat([inputs_re, inputs_im], axis=4)
        complex_flag = True
        shapes.append(2)
    elif rank == 5 and shapes[-1] == 2:
        complex_flag = False
        pass
    else:
        raise TypeError('Check input tensor dtypes or shape')

    conv = tf.transpose(inputs, perm=[0, 1, 2, 4, 3])
    if type(kernal) is int:
        kernal_size = (kernal,kernal,1)
    elif type(kernal) is tuple and len(kernal) == 2:
        kernal_size = kernal + (1,)
    else:
        raise NameError('Unacceptable Kernal Size')

    if type(strides) is int:
        strides = (strides,strides,1)
    elif type(strides) is tuple and len(strides) == 2:
        strides = strides + (1,)
    else:
        raise NameError('Unacceptable Kernal Size')

    conv = tf.layers.conv3d_transpose(conv, filters * 2, kernal_size, strides=strides, padding=padding)
    shapes_conv = conv.get_shape().as_list()
    conv = tf.reshape(conv, [-1, shapes_conv[1], shapes_conv[2], shapes_conv[3] * 2, filters])
    assert (shapes[4] == 2)
    conv_re = conv[:, :, :, 0, :] - conv[:, :, :, 3, :]
    conv_im = conv[:, :, :, 1, :] - conv[:, :, :, 2, :]
    conv_re = tf.reshape(conv_re, [-1, shapes_conv[1], shapes_conv[2], 1, filters])
    conv_im = tf.reshape(conv_im, [-1, shapes_conv[1], shapes_conv[2], 1, filters])
    output = tf.concat([conv_re, conv_im], axis=3)
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
    if complex_flag:
        output = tf.complex(output[:, :, :, :, 0], output[:, :, :, :, 1])

    return output
