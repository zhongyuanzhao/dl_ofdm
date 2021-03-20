#!/usr/bin/env python
######################################################################################
# Library for wireless channel emulator and radio related functions
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
import scipy as sp
from contextlib import closing
import multiprocessing as mp
import os
# from sklearn.preprocessing import OneHotEncoder
from complex import *
import copy



def _init(shared_arr_, shared_arr2_, inputs_):
    # The shared array pointer is a global variable so that it can be accessed by the
    # child processes. It is a tuple (pointer, dtype, shape).
    global shared_arr
    shared_arr = shared_arr_
    global shared_arr2
    shared_arr2 = shared_arr2_
    global shared_input
    shared_input = inputs_


def shared_to_numpy(shared_arr, dtype, shape):
    """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
    No copy is involved, the array reflects the underlying shared buffer."""
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


def create_shared_array(dtype, shape):
    """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
    Note that the buffer values are not initialized.
    """
    dtype = np.dtype(dtype)
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    # Create the RawArray instance.
    shared_arr = mp.RawArray(cdtype, sum(shape))
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr



def AWGN_channel(inputs, SNR):
    '''
    AWGN channel simulator
    :param inputs: input tensor
    :param SNR: SNR of signal
    :return:
    '''
    shapes = inputs.get_shape()
    assert(shapes[-1]==2)
    input_real = inputs[:,:,:,0:1]
    batch_mean2, batch_var2 = tf.nn.moments(inputs, [0])
    inputs = tf.nn.batch_normalization(inputs, batch_mean2, batch_var2, offset=None, scale=None,
                                         variance_epsilon=1e-8) / np.sqrt(2.0)
    # power_tx = tf.reduce_mean(tf.square(tf.norm(inputs, axis=-1)))
    level = np.sqrt(.5)*tf.pow(10.0, tf.negative(SNR)/20.0) # Re and Im Max Value of power 1 signal is sqrt(1/2)
    noise_phase = tf.random_uniform(tf.shape(input_real), maxval=2*np.pi, dtype=tf.float32)
    noise_amp = tf.multiply(tf.reshape(level,[-1,1,1,1]), tf.random_normal(tf.shape(input_real), stddev=1.0, dtype=tf.float32))
    noise_real = tf.multiply(tf.abs(noise_amp), tf.math.sin(noise_phase))
    noise_imag = tf.multiply(tf.abs(noise_amp), tf.math.cos(noise_phase))
    # print(noise_level)
    noise = tf.concat([noise_real, noise_imag], axis=-1)
    # noise = tf.multiply(tf.reshape(noise_amp,[-1,1,1,1]), tf.random_normal(tf.shape(inputs), stddev=1.0))
    y_noisy = inputs + noise
    noise_power = tf.reduce_mean(tf.square(noise[:,:,:,0])+tf.square(noise[:,:,:,1]))
    # noise_amp = tf.norm(noise, axis=-1)
    # noise_power = tf.reduce_mean(tf.square(noise_amp))
    return y_noisy, noise_power


def AWGN_channel_norm(inputs, SNR, norm):
    '''
    AWGN channel simulator
    :param inputs: input tensor
    :param SNR: tensor SNR of signal
    :param norm: scalar condition tensor (int32) to turn on/off noise normalization by inputs
    :return:
    '''
    shapes = inputs.get_shape()
    assert(shapes[-1]==2)
    N_signal = shapes[1].value*shapes[2].value
    batch_mean2, batch_var2 = tf.nn.moments(inputs, [0])
    inputs = tf.nn.batch_normalization(inputs, batch_mean2, batch_var2, offset=None, scale=None,
                                         variance_epsilon=1e-6) / np.sqrt(2)
    # power_tx = tf.reduce_mean(tf.square(tf.norm(inputs, axis=-1)))
    signal_amp = tf.norm(inputs, axis=-1)
    signal_amp = tf.reshape(signal_amp, [-1, N_signal])
    signal_power = tf.reduce_mean(tf.square(signal_amp), axis=1, keepdims=True)
    avg_signal_amp = tf.sqrt(signal_power)

    noise_level = np.sqrt(1/2.) * tf.pow(10.0, tf.negative(SNR)/20.0)  # Re and Im Max Value of power 1 signal is sqrt(1/2)
    noise_level = tf.cond(norm > 0, lambda: tf.multiply(noise_level, avg_signal_amp), lambda: noise_level)

    # print(noise_level)
    noise = tf.multiply(tf.reshape(noise_level,[-1,1,1,1]), tf.random_normal(tf.shape(inputs), stddev=1.0))
    y_noisy = inputs + noise
    # noise_power = tf.reduce_mean(tf.square(noise[:,:,:,0])+tf.square(noise[:,:,:,1]))
    noise_amp  = tf.norm(noise, axis=-1)
    noise_power = tf.reduce_mean(tf.square(noise_amp))
    return y_noisy, noise_power


def RayLeigh_channel(inputs, SNR, n_tap=8, chan='ETU', samp_rate=1e7):
    '''
    RayLeigh multipath fading channel simulator
    :param inputs: input tensor
    :param SNR: SNR
    :param n_tap: number of taps
    :param chan: 'ETU', 'EPA', 'EVA' LTE multipath channels
    :return:
    '''
    shapes = inputs.get_shape()
    # power_tx = tf.reduce_mean(tf.square(tf.norm(inputs, axis=-1)))
    # print(noise_level)

    N_signal = shapes[1].value*shapes[2].value
    assert(shapes[-1]==2)
    if chan == 'ETU':
        tap_delay = [0, 50, 120, 200, 230, 500, 1600, 2300, 5000]
        tap_powdB = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0]
    elif chan == 'EPA':
        tap_delay = [0, 30, 70, 90, 110, 190, 410]
        tap_powdB = [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8]
    elif chan == 'EVA':
        tap_delay = [0, 30, 150, 310, 370, 710, 1090, 1730, 2510]
        tap_powdB = [0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9]
    elif n_tap == 1:
        tap_delay = [0]
        tap_powdB = [0]
    else:
        raise ValueError('Unknown channel type.')
    tap_delay = np.asarray(tap_delay)
    tap_powdB = np.asarray(tap_powdB)

    T_ns = 1e9/samp_rate
    N_fir = np.minimum(int(np.ceil(tap_delay[-1]/T_ns))+1, N_signal)
    ch_coeff = np.zeros([N_fir,2])
    c_tap = np.ceil(tap_delay/T_ns).astype(int)
    c_taps, c_idx = np.unique(c_tap, return_index=True)
    c_powdB = tap_powdB[c_idx]/10.0
    c_pow = 10.0 ** c_powdB
    n_tap = np.sum(c_pow)#len(c_pow)
    ch_coeff[c_taps,0] = c_pow * (1/np.sqrt(n_tap))
    ch_coeff[c_taps,1] = c_pow * (1/np.sqrt(n_tap))
    # ht = (1/np.sqrt(2)) * (1/np.sqrt(n_tap)) * tf.random_normal([n_tap,2], dtype=tf.complex64, stddev=1.0)
    ht = tf.random_normal([N_fir,2], dtype=tf.float32, stddev=1.0/np.sqrt(2))
    ht = tf.multiply(ht, ch_coeff)
    ht = tf.reshape(ht, [N_fir, 1, 1, 2])
    inputs_complex = tf.reshape(inputs, [-1, shapes[1]*shapes[2], 1, shapes[3]])
    # convolution
    inputs_chan_res = nn_conv1d_complex(inputs_complex, ht)
    # convert back to real format
    inputs_chan_res = tf.reshape(inputs_chan_res, tf.shape(inputs))
    signal_amp = tf.norm(inputs_chan_res, axis=-1)
    signal_amp = tf.reshape(signal_amp, [-1, N_signal])
    distorted_signal_power = tf.reduce_mean(tf.square(signal_amp), axis=1, keepdims=True)
    distorted_signal_amp = tf.sqrt(distorted_signal_power)

    # noise_level = np.sqrt(1/2.) * tf.pow(10.0, tf.negative(SNR)/20.0) # Re and Im Max Value of power 1 signal is sqrt(1/2)
    noise_level = np.sqrt(1/2.) * tf.pow(10.0, tf.negative(SNR)/20.0)  # Re and Im Max Value of power 1 signal is sqrt(1/2)
    noise_level = tf.multiply(noise_level, distorted_signal_amp)
    noise = tf.multiply(tf.reshape(noise_level,[-1,1,1,1]), tf.random_normal(tf.shape(inputs), stddev=1.0))
    # Add Noise
    y_noisy = inputs_chan_res + noise
    # noise_power = tf.reduce_mean(tf.square(noise[:,:,:,0])+tf.square(noise[:,:,:,1]))
    noise_amp = tf.norm(noise, axis=-1)
    noise_power = tf.reduce_mean(tf.square(noise_amp))
    return y_noisy, noise_power


## Next is the numpy version of AWGN and Rayleigh Channel


def RayLeigh_channel_np(inputs, FLAGS, samp_rate=0.96e6):
    '''
    RayLeigh multipath fading channel simulator
    :param inputs: input ndarray type complex
    :param chan: 'AWGN', 'ETU', 'EPA', 'EVA' LTE multipath channels
    :param samp_rate: sample rate of signal
    :return:
    '''
    assert(np.iscomplexobj(inputs))
    shapes = inputs.shape
    n_fr, n_sym, n_sc = shapes
    nSymbol = FLAGS.nsymbol #8 # number of OFDM symbols per frame
    chan = FLAGS.channel
    N_signal = shapes[1]*shapes[2]
    n_tap = 8
    H_fr = np.zeros([n_fr, n_sym, FLAGS.nfft], dtype=np.complex64)
    if chan.lower() == 'awgn':
        y_distored = inputs
        H_fr = np.ones([n_fr, n_sym, FLAGS.nfft], dtype=np.complex64)
    else:
        if chan.lower() == 'etu':
            tap_delay = [0, 50, 120, 200, 230, 500, 1600, 2300, 5000]
            tap_powdB = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0]
        elif chan.lower() == 'epa':
            tap_delay = [0, 30, 70, 90, 110, 190, 410]
            tap_powdB = [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8]
        elif chan.lower() == 'eva':
            tap_delay = [0, 30, 150, 310, 370, 710, 1090, 1730, 2510]
            tap_powdB = [0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9]
        else:
            tap_delay = [0]
            tap_powdB = [0]
            n_tap = 1

        tap_delay = np.asarray(tap_delay)
        tap_powdB = np.asarray(tap_powdB)

        T_ns = 1e9/samp_rate
        N_fir = np.minimum(int(np.ceil(tap_delay[-1]/T_ns))+1, N_signal)
        ch_coeff = np.zeros([N_fir,2])
        c_tap = np.ceil(tap_delay/T_ns).astype(int)
        c_taps, c_idx = np.unique(c_tap, return_index=True)
        c_powdB = tap_powdB[c_idx]/10.0
        c_pow = 10.0 ** c_powdB
        n_tap = np.sum(c_pow)#len(c_pow)
        ch_coeff[c_taps,0] = c_pow * (1/np.sqrt(n_tap))
        ch_coeff[c_taps,1] = c_pow * (1/np.sqrt(n_tap))
        # ht = (1/np.sqrt(2)) * (1/np.sqrt(n_tap)) * tf.random_normal([n_tap,2], dtype=tf.complex64, stddev=1.0)

        y_distored = np.zeros(inputs.shape,dtype=np.complex64)
        for i_fr in range(n_fr):
            ht = np.random.normal(loc=0.0, scale=1.0/np.sqrt(2), size=[N_fir,2])
            ht = np.multiply(ht, ch_coeff)
            ht = ht[:,0] + ht[:,1]*1j
            tx_signal = np.reshape(inputs[i_fr,:,:],[n_sym*n_sc,])
            rx_signal = np.convolve(tx_signal, ht, mode='same')
            y_distored[i_fr,:,:] = rx_signal.reshape([n_sym, n_sc])
            Ht = np.fft.fft(ht,FLAGS.nfft)
            H_fr[i_fr, :, :] = Ht

    y_real = np.reshape(np.real(y_distored),[n_fr, n_sym, n_sc, 1])
    y_imag = np.reshape(np.imag(y_distored),[n_fr, n_sym, n_sc, 1])
    y_out = np.concatenate([y_real,y_imag], axis=-1)
    return y_out, H_fr


def doppler_inner(shared_input, i):
    t_sym, const1, nfft, n_sc, f_kn_re, f_kn_im, theta_kn_re, theta_kn_im, ch_coeff, alpha_matrix, tx_signal = shared_input
    var_t = i * t_sym
    tmp_re = np.cos(2 * np.pi * var_t * f_kn_re + theta_kn_re)
    tmp_im = np.cos(2 * np.pi * var_t * f_kn_im + theta_kn_im)
    mu_re = const1 * np.sum(tmp_re, 0)
    mu_im = const1 * np.sum(tmp_im, 0)
    zck = mu_re + mu_im * 1j
    # zck = zcks[:, i]
    a_taps = np.multiply(zck, ch_coeff)
    gt = np.matmul(a_taps, alpha_matrix)
    rx_signal_roll = np.convolve(tx_signal, gt, mode='same')
    rx_signal_sym = rx_signal_roll[n_sc * i:n_sc * (i + 1)]
    ht_sym = np.fft.fft(gt, nfft)
    return rx_signal_sym, ht_sym


class rayleigh_chan_lte:
    '''
    RayLeigh multipath fading channel simulator, see https://www.mathworks.com/help/comm/ug/fading-channels.html
    Doppler model: Jake's model
    Fading technique: sum of sinusoids approach
    :param inputs: input ndarray type complex
    :param chan: 'AWGN', 'ETU', 'EPA', 'EVA' LTE multipath channels
    :param samp_rate: sample rate of signal
    '''
    def __init__(self, FLAGS, sample_rate=0.96e6, mobile=False, mix=False):
        self.nSymbol = FLAGS.nsymbol
        self.chan = FLAGS.channel.lower()
        self.sample_rate=sample_rate
        self.nfft = FLAGS.nfft
        self.T_ns = 1e9 / self.sample_rate
        self.Fd = 0.0
        self.ss = 48 # number of sinusoids
        self.const1 = np.sqrt(1.0 / self.ss)
        self.mobile = mobile
        self.mix = mix # mix Doppler and no Doppler in alternative frames
        self.cpu_count = mp.cpu_count()
        # self.pool = mp.Pool(processes=7)
        if self.chan == 'mixrayleigh':
            self.tap_powdB_list = []
            self.tap_delay_list = []
            self.n_taps_list = []
            self.ch_coeff_list = []
            self.alpha_matrix_list = []
            self.Fd_list = []
            for chan in ['flat', 'etu', 'eva', 'epa']:
                self.chan = chan
                self.load_tap()
                self.load_alpha_matrix()
                self.tap_powdB_list.append(self.tap_powdB)
                self.tap_delay_list.append(self.tap_delay)
                self.n_taps_list.append(self.n_taps)
                self.ch_coeff_list.append(self.ch_coeff)
                self.alpha_matrix_list.append(self.alpha_matrix)
                self.Fd_list.append(self.Fd)
            self.chan = 'mixrayleigh'
        elif self.chan == 'mixall':
            self.tap_powdB_list = []
            self.tap_delay_list = []
            self.n_taps_list = []
            self.ch_coeff_list = []
            self.alpha_matrix_list = []
            self.Fd_list = []
            for chan in ['awgn', 'flat', 'etu', 'eva', 'epa']:
                self.chan = chan
                self.load_tap()
                self.load_alpha_matrix()
                self.tap_powdB_list.append(self.tap_powdB)
                self.tap_delay_list.append(self.tap_delay)
                self.n_taps_list.append(self.n_taps)
                self.ch_coeff_list.append(self.ch_coeff)
                self.alpha_matrix_list.append(self.alpha_matrix)
                self.Fd_list.append(self.Fd)
            self.chan = 'mixall'
        else:
            self.load_tap()
            self.load_alpha_matrix()

    def load_tap(self):
        if self.chan == 'etu':
            self.tap_delay = np.asarray([0, 50, 120, 200, 230, 500, 1600, 2300, 5000])
            self.tap_powdB = np.asarray([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0])
            if self.mobile:
                self.Fd = 300.0
        elif self.chan == 'epa':
            self.tap_delay = np.asarray([0, 30, 70, 90, 110, 190, 410])
            self.tap_powdB = np.asarray([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
            if self.mobile:
                self.Fd = 5.0
        elif self.chan == 'eva':
            self.tap_delay = np.asarray([0, 30, 150, 310, 370, 710, 1090, 1730, 2510])
            self.tap_powdB = np.asarray([0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])
            if self.mobile:
                self.Fd = 70.0
        elif self.chan == 'custom':
            self.tap_delay = np.asarray([0, 70, 200, 230, 500, 1600, 2700, 3000])
            self.tap_powdB = np.asarray([0.0, -1.4, -1.4, -1.0, -3.0, -9.1, -15.0, -19.0])
            if self.mobile:
                self.Fd = 80.0
        else:
            self.tap_delay = np.asarray([0])
            self.tap_powdB = np.asarray([0])
            if self.mobile:
                self.Fd = 5.0
            else:
                self.Fd = 0.0
        self.n_taps = np.size(self.tap_delay)
        c_powdB = self.tap_powdB / 10.0
        c_pow = 10.0 ** c_powdB
        p_taps = np.sum(c_pow)  # len(c_pow)
        self.ch_coeff = c_pow * (1 / np.sqrt(p_taps))
        return

    def load_alpha_matrix(self):
        if self.chan == 'etu':
            self.alpha_matrix = np.genfromtxt('./3gpp/AM_ETU.csv', delimiter=',')
        elif self.chan == 'epa':
            self.alpha_matrix = np.genfromtxt('./3gpp/AM_EPA.csv', delimiter=',')
        elif self.chan == 'eva':
            self.alpha_matrix = np.genfromtxt('./3gpp/AM_EVA.csv', delimiter=',')
        elif self.chan == 'custom':
            self.alpha_matrix = np.genfromtxt('./3gpp/AM_Custom.csv', delimiter=',')
        else:
            self.alpha_matrix = np.ones([1, 1], dtype=np.float64)
        return

    def doppler_realize(self, Fd, n_taps):
        k_vec = np.arange(1, n_taps + 1)
        n_vec = (np.arange(1, self.ss + 1).reshape((self.ss,1)) - 0.5) * np.pi / (4*self.ss)
        alpha_k0_re = k_vec * np.pi / (4*self.ss)
        alpha_k0_im = - alpha_k0_re
        f_kn_re = Fd * np.cos(np.add(n_vec, alpha_k0_re))
        f_kn_im = Fd * np.cos(np.add(n_vec, alpha_k0_im))
        theta_kn_re = np.random.uniform(0, 2*np.pi, size=(self.ss, n_taps))
        theta_kn_im = np.random.uniform(0, 2*np.pi, size=(self.ss, n_taps))
        return f_kn_re, f_kn_im, theta_kn_re, theta_kn_im


    def doppler_channel(self, tx_signal, Fd, ch_coeff, alpha_matrix, n_taps, n_sym, n_sc):
        rx_signal = np.zeros_like(tx_signal)
        # rx_signal = np.zeros((n_sc, n_sym), dtype=np.complex64)
        tx_signal_pre = np.zeros(shape=(n_taps + n_sym*n_sc,), dtype=np.complex64)
        tx_signal_pre[n_taps:] = tx_signal
        f_kn_re, f_kn_im, theta_kn_re, theta_kn_im = self.doppler_realize(Fd, n_taps)
        Ht = np.zeros((n_sym, self.nfft), dtype=np.complex64)
        t_sym = n_sc / self.sample_rate

        for i in range(n_sym):
            var_t = i * t_sym
            tmp_re = np.cos(2 * np.pi * var_t * f_kn_re + theta_kn_re)
            tmp_im = np.cos(2 * np.pi * var_t * f_kn_im + theta_kn_im)
            mu_re = self.const1 * np.sum(tmp_re, 0)
            mu_im = self.const1 * np.sum(tmp_im, 0)
            zck = mu_re + mu_im*1j
            # zck = zcks[:, i]
            a_taps = np.multiply(zck, ch_coeff)
            gt = np.matmul(a_taps, alpha_matrix)
            tx_signal_roll = tx_signal_pre[n_sc*i: n_taps+n_sc*(i+1)]
            rx_signal_roll = np.convolve(tx_signal_roll, gt, mode='same')
            rx_signal[n_sc*i:n_sc*(i+1)] = rx_signal_roll[n_taps:]
            Ht[i, :] = np.fft.fft(gt, self.nfft)
        return rx_signal, Ht

    def channel(self, tx_signal, Fd, ch_coeff, alpha_matrix, n_taps, n_sym, n_sc, doppler=False):
        if doppler:
            rx_signal, Ht = self.doppler_channel(tx_signal, Fd,
                                                 ch_coeff,
                                                 alpha_matrix,
                                                 n_taps,
                                                 n_sym, n_sc)
        else:
            zrk = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(2), size=[n_taps, 2])
            zck = zrk[:, 0] + zrk[:, 1] * 1j
            a_taps = np.multiply(zck, ch_coeff)
            gt = np.matmul(a_taps, alpha_matrix)
            rx_signal = np.convolve(tx_signal, gt, mode='same')
            Ht = np.fft.fft(gt, self.nfft)
        return rx_signal, Ht

    def run(self, inputs):
        assert(np.iscomplexobj(inputs))
        shapes = inputs.shape
        n_fr, n_sym, n_sc = shapes
        n_samples_fr = n_sym * n_sc
        # ch_gains = np.zeros([n_fr, 1])
        H_fr = np.zeros([n_fr, n_sym, self.nfft], dtype=np.complex64)
        if self.chan == 'awgn':
            y_distored = inputs
            H_fr = np.ones([n_fr, n_sym, self.nfft], dtype=np.complex64)
        elif self.chan == 'mixrayleigh':
            y_distored = np.zeros(inputs.shape, dtype=np.complex64)
            for i_fr in range(n_fr):
                tx_signal = np.reshape(inputs[i_fr, :, :], [n_samples_fr, ])
                fr_sel = i_fr % 4
                n_taps = self.n_taps_list[fr_sel]
                ch_coeff = self.ch_coeff_list[fr_sel]
                alpha_matrix = self.alpha_matrix_list[fr_sel]
                Fd = self.Fd_list[fr_sel]
                dp_sel = i_fr % 3
                doppler = (dp_sel == 0) and (Fd > 0.1) and self.mix

                rx_signal, Ht = self.channel(tx_signal,
                                             Fd, ch_coeff, alpha_matrix, n_taps,
                                             n_sym, n_sc,
                                             doppler=doppler)
                y_distored[i_fr, :, :] = rx_signal.reshape([n_sym, n_sc])
                H_fr[i_fr, :, :] = Ht
        elif self.chan == 'mixall':
            y_distored = np.zeros(inputs.shape, dtype=np.complex64)
            for i_fr in range(n_fr):
                tx_signal = np.reshape(inputs[i_fr, :, :], [n_samples_fr, ])
                fr_sel = i_fr % 5
                if fr_sel == 0:
                    rx_signal = tx_signal
                    gt = np.array([1+0j])
                    Ht = np.fft.fft(gt, self.nfft)
                else:
                    n_taps = self.n_taps_list[fr_sel]
                    ch_coeff = self.ch_coeff_list[fr_sel]
                    alpha_matrix = self.alpha_matrix_list[fr_sel]
                    Fd = self.Fd_list[fr_sel]
                    dp_sel = i_fr % 4
                    doppler = (dp_sel == 0) and (Fd > 0.1) and self.mix

                    rx_signal, Ht = self.channel(tx_signal,
                                                 Fd, ch_coeff, alpha_matrix, n_taps,
                                                 n_sym, n_sc,
                                                 doppler=doppler)
                y_distored[i_fr, :, :] = rx_signal.reshape([n_sym, n_sc])
                H_fr[i_fr, :, :] = Ht
        else:
            y_distored = np.zeros(inputs.shape, dtype=np.complex64)
            for i_fr in range(n_fr):
                tx_signal = np.reshape(inputs[i_fr, :, :], [n_samples_fr, ])
                rx_signal, Ht = self.channel(tx_signal,
                                             self.Fd, self.ch_coeff, self.alpha_matrix, self.n_taps,
                                             n_sym, n_sc,
                                             doppler=(self.Fd > 0.1))
                y_distored[i_fr, :, :] = rx_signal.reshape([n_sym, n_sc])
                H_fr[i_fr, :, :] = Ht

        y_real = np.reshape(np.real(y_distored), [n_fr, n_sym, n_sc, 1])
        y_imag = np.reshape(np.imag(y_distored), [n_fr, n_sym, n_sc, 1])
        y_out = np.concatenate([y_real, y_imag], axis=-1)
        # mean_zck_pwr = np.mean(ch_gains) # analysis
        return y_out, H_fr

    def __call__(self, inputs):
        y_out, H_fr = self.run(inputs)
        return y_out, H_fr


def AWGN_channel_np(inputs, SNR):
    shapes = inputs.shape
    # sig_pwr = np.linalg.norm(inputs,axis=-1) + 0.0000001
    sig_pwr = np.square(inputs[:,:,:,0:1]) + np.square(inputs[:,:,:,1:])
    savg_pwr = np.nanmean(sig_pwr)
    inputs_norm = inputs/np.sqrt(savg_pwr)
    noise = np.random.randn(shapes[0], shapes[1], shapes[2], shapes[3])
    noisestd = np.sqrt(0.5)*np.power(10.0,(-SNR/20.0))
    noisestd = np.reshape(noisestd, [-1,1,1,1])
    noise = noise * noisestd
    outputs = inputs_norm + noise
    noise_power = np.square(noise[:,:,:,0:1]) + np.square(noise[:,:,:,1:])
    noise_power = np.mean(noise_power)
    return outputs, noise_power


