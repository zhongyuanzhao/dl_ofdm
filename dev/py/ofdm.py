#!/usr/bin/env python
######################################################################################
# OFDM system implemented in python NumPy
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


# mu = 4 # bits per symbol (i.e. 16QAM)
# payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

mapping_16qam = {
    (0,0,0,0) : -3+3j,
    (1,0,0,0) : -3+1j,
    (0,1,0,0) : -3-3j,
    (1,1,0,0) : -3-1j,
    (0,0,1,0) : -1+3j,
    (1,0,1,0) : -1+1j,
    (0,1,1,0) : -1-3j,
    (1,1,1,0) : -1-1j,
    (0,0,0,1) :  3+3j,
    (1,0,0,1) :  3+1j,
    (0,1,0,1) :  3-3j,
    (1,1,0,1) :  3-1j,
    (0,0,1,1) :  1+3j,
    (1,0,1,1) :  1+1j,
    (0,1,1,1) :  1-3j,
    (1,1,1,1) :  1-1j
}

mapping_qpsk = {
    (0,0) : -3+3j,
    (1,0) : -3-3j,
    (0,1) :  3+3j,
    (1,1) :  3-3j
}

mapping_bpsk = {
    (0) : -4.24264+0.0j,
    (1) :  4.24264+0.0j,
}

#const1 = 3./np.sqrt(2.)=2.12132
#const1 = 3.*np.sqrt(2.)=4.24264
mapping_8psk = {
    (0,0,0) : -3-3j,
    (0,0,1) : -4.24264+0j,
    (0,1,0) :  0+4.24264j,
    (0,1,1) : -3+3j,
    (1,0,0) : -4.24264+0j,
    (1,0,1) :  3-3j,
    (1,1,0) :  3+3j,
    (1,1,1) :  4.24264+0j
}

#scale: abs(3+3i)/abs(3+1i) = 4.2426/3.1623
mapping_8qam = {
    (0,0,0) : (-3+1j)*(4.2426/3.1623),
    (1,0,0) : (-3-1j)*(4.2426/3.1623),
    (0,1,0) : (-1+1j)*(4.2426/3.1623),
    (1,1,0) : (-1-1j)*(4.2426/3.1623),
    (0,0,1) : ( 3+1j)*(4.2426/3.1623),
    (1,0,1) : ( 3-1j)*(4.2426/3.1623),
    (0,1,1) : ( 1+1j)*(4.2426/3.1623),
    (1,1,1) : ( 1-1j)*(4.2426/3.1623)
}

'''
tf.complex
tf.conj
tf.imag
tf.angle
tf.real

tf.abs
tf.multiply
'''
def constellation_map(ord = 1):
    '''
    N-dimentional ndarray of constellation complex number
    :param ord: N
    :return: ndarray
    '''
    assert(ord > 0)
    assert(ord < 5)
    dims = [2]*ord
    maps = np.empty(dims, dtype=np.complex64)
    if ord == 1:
        for i0 in range(2):
            maps[i0] = mapping_bpsk[(i0)]
    elif ord == 2:
        for i0 in range(2):
            for i1 in range(2):
                maps[i0,i1] = mapping_qpsk[(i0,i1)]
    elif ord == 3:
        for i0 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    maps[i0,i1,i2] = mapping_8qam[(i0,i1,i2)]
    elif ord==4:
        for i0 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    for i3 in range(2):
                        maps[i0,i1,i2,i3] = mapping_16qam[(i0, i1, i2, i3)]
    return maps


def const_map(ord = 1):
    '''
    N-dimentional ndarray of constellation complex number
    :param ord: N
    :return: ndarray
    '''
    assert(ord > 0)
    assert(ord < 5)
    dims = 2**ord
    maps = np.empty([dims,], dtype=np.complex64)
    i = 0
    if ord == 1:
        for i0 in range(2):
            maps[i0] = mapping_bpsk[(i0)]
    elif ord == 2:
        for i0 in range(2):
            for i1 in range(2):
                maps[i] = mapping_qpsk[(i0,i1)]
                i += 1
    elif ord == 3:
        for i0 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    maps[i] = mapping_8qam[(i0,i1,i2)]
                    i += 1
    elif ord == 4:
        for i0 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    for i3 in range(2):
                        maps[i] = mapping_16qam[(i0, i1, i2, i3)]
                        i += 1
    return maps


def Clip_by_norm_np(inputs, peak=8.0):
    '''
    Clip complex number by PAPR
    :param inputs: complex matrix
    :param peak:
    :return:
    '''
    assert(inputs.dtype==np.complex)
    sig_pwr = np.square(np.abs(inputs)) + 1.0e-8
    pwr_shapes = inputs.shape
    sig_pwr = np.reshape(sig_pwr,pwr_shapes)
    savg_pwr = np.nanmean(sig_pwr)
    clip_val = np.sqrt(peak) * inputs / np.sqrt(sig_pwr)
    outputs = np.where(sig_pwr < peak*savg_pwr, inputs, clip_val)
    return outputs


def get_lte_dl_cfg(nFFT):
    '''
    Calculate sample rate based on LTE downlink configuration, 1
    :param nFFT:
    :return: sample rate, i.e. 0.96e6 (0.96Msps)
    '''
    sample_rate_dict = {64: 0.96e6,
                        128: 1.92e6,
                        256: 3.84e6,
                        512: 7.68e6,
                        1024: 15.36e6,
                        1536: 23.04e6,
                        2048: 30.72e6}
    nrb_dict = {64: 4,
                128: 8,
                256: 15,
                512: 25,
                1024: 50,
                1536: 75,
                2048: 100}
    assert(nFFT in sample_rate_dict.keys())
    return sample_rate_dict[nFFT], nrb_dict[nFFT]



class ofdm_tx:
    def __init__(self, FLAGS):
        self.nSymbol = FLAGS.nsymbol  # 8 # number of OFDM symbols per frame
        self.K = FLAGS.nfft  # 64  # number of OFDM subcarriers
        if FLAGS.longcp:
            self.CP = int(np.around(self.K * 0.25))  # length of the cyclic prefix: 25% of the block
        else:
            self.CP = int(np.around(self.K * 0.07))
        self.Fs, self.nRB = get_lte_dl_cfg(self.K)
        self.DC = 2
        if FLAGS.pilot == 'lte':
            self.P = 2*self.nRB
            self.G = self.K - self.DC - self.nRB * 12
        else:
            self.P = FLAGS.npilot # number of pilot carriers per OFDM symbol
            self.G = FLAGS.nguard # number of guard subcarriers per OFDM symbol
        self.pilotValue = 3 + 3j  # The known value each pilot transmits
        self.guardValue = 0
        self.nbits = FLAGS.nbits

        # For Comb Type Pilot
        self.allCarriers = np.arange(self.K)  # indices of all subcarriers ([0, 1, ... K-1])
        self.DCCarriers = np.arange(self.K // 2 - 1, self.K // 2 + 1, dtype=np.int32)
        self.effecCarriers = np.arange(self.G // 2, self.K - self.G // 2)
        self.effecCarriers = np.setdiff1d(self.effecCarriers, self.DCCarriers)
        self.pilot_loc = np.arange(0,len(self.effecCarriers), int(np.ceil(float(len(self.effecCarriers)) / self.P)))
        self.pilotCarriers = self.effecCarriers[self.pilot_loc]  # Pilots is every (K/P)th carrier.
        self.guardCarriers = np.setdiff1d(self.allCarriers, self.effecCarriers) # Guard Sc include DC
        # data carriers are all remaining carriers
        self.dataCarriers = np.setdiff1d(self.effecCarriers, self.pilotCarriers)

        self.allSc = np.arange(self.K * self.nSymbol)   # number all sc in a frame
        self.effecSc = np.empty([(self.K - self.G - self.DC), self.nSymbol], dtype=int)
        if FLAGS.pilot == 'scattered':
            # For Scattered Pilot in a Frame
            self.pilotSc = np.empty([len(self.pilotCarriers), self.nSymbol], dtype=int)
            for idx in range(self.nSymbol):
                self.effecSc[:, idx] = self.effecCarriers + idx * self.K
                pilot_loc = np.sort((self.pilot_loc + idx * 3) % len(self.effecCarriers))
                self.pilotSc[:, idx] = self.effecCarriers[pilot_loc] + idx * self.K
        elif FLAGS.pilot == 'block':
            # For Block Type Pilot in a Frame
            self.pilotSc = np.empty([len(self.effecCarriers), 1], dtype=int)
            for idx in range(self.nSymbol):
                self.effecSc[:, idx] = self.effecCarriers + idx * self.K
                if idx == 3:
                    pilot_loc = np.arange(0, len(self.effecCarriers), 1, dtype=int)
                    self.pilotSc[:, idx//4] = self.effecCarriers[pilot_loc] + idx * self.K
        elif FLAGS.pilot == 'comb':
            # For Block Type Pilot in a Frame
            self.pilotSc = np.empty([len(self.effecCarriers), 2], dtype=int)
            for idx in range(self.nSymbol):
                self.effecSc[:, idx] = self.effecCarriers + idx * self.K
                self.pilotSc[:, idx] = self.effecCarriers[self.pilot_loc] + idx * self.K
        elif FLAGS.pilot == 'lte':
            # For Scattered Pilot in a Frame
            assert(self.nSymbol == 7)
            self.pilotSc = np.empty([self.P, 2], dtype=int)
            for idx in range(self.nSymbol):
                self.effecSc[:, idx] = self.effecCarriers + idx * self.K
                if idx == 0:
                    pilot_loc = np.sort(self.pilot_loc % len(self.effecCarriers))
                    self.pilotSc[:, 0] = self.effecCarriers[pilot_loc] + idx * self.K
                elif idx == 4:
                    pilot_loc = np.sort((self.pilot_loc + 3) % len(self.effecCarriers))
                    self.pilotSc[:, 1] = self.effecCarriers[pilot_loc] + idx * self.K
        else:
            raise ValueError('Unsupported pilot type %s.'%FLAGS.pilot)
        self.effecSc = self.effecSc.reshape((-1, ), order='F')
        self.pilotSc = self.pilotSc.reshape((-1, ), order='F')
        self.pilotSc = np.sort(self.pilotSc)
        self.guardSc = np.setdiff1d(self.allSc, self.effecSc)
        self.dataSc = np.setdiff1d(self.effecSc, self.pilotSc)

        self.frame_size = len(self.dataSc)
        self.pilot_size = len(self.pilotSc)


    def ofdm_tx_np(self, inputs):
        '''
        There will be no NN layers in this transmitter, only conversion
        Add frame based pilot position routation
        This will be outside the TF graph, generate data
        :param inputs: numpy array, bits, will be labels
        :return: OFDM data
        '''
        n_sym, n_sc, nbits = inputs.shape
        n_sym, n_sc, nbits = int(n_sym), int(n_sc), int(nbits)
        n_sc_cp = self.K + self.CP
        n_frame = n_sym//self.nSymbol
        assert(n_sc+self.P+self.G+self.DC == self.K)
        assert(nbits < 5)

        # Step 1, Real-2-Complex, Constellation Mapping
        const_table = const_map(nbits)
        bits = inputs.reshape([-1, nbits])
        bits.astype(np.int32)
        bits = np.pad(bits,[(0,0),(8-nbits,0)],mode='constant')
        sym_dec = np.packbits(bits, axis=1)
        # cmpx_iq = [const_table[tuple(row)] for row in bits]
        cmpx_iq = const_table.take(sym_dec) # Vectorize alternative, reduce execution time from 18 sec to <1 sec
        cmpx_iq = np.reshape(cmpx_iq, [-1, n_sc])

        # Step 2, Padding Zeros at Guard SC and DC; haven't remove DC yet
        # This code for scattered pilot pattern
        symbol = np.zeros([n_sym, self.K], dtype=np.complex64)
        symbol[:, self.dataCarriers] = cmpx_iq
        symbol[:, self.pilotCarriers] = self.pilotValue

        # Step 3: IFFT
        ofdm_time = np.fft.ifft(symbol)
        # GPU accelerated version
        # x = gpuarray.to_gpu(symbol)
        # y = gpuarray.empty(symbol.shape, np.complex64)
        # ofdm_time = fft_scikit(x, y)

        # Step 4: Add CP
        ofdm_CP = ofdm_time[:,-self.CP:]
        ofdm_symbol_cpx = np.concatenate([ofdm_CP,ofdm_time],axis=1)
        ofdm_symbol_cpx = ofdm_symbol_cpx.reshape([-1, self.nSymbol, n_sc_cp])
        ofdm_symbol_re = np.reshape(np.real(ofdm_symbol_cpx),[-1, self.nSymbol, n_sc_cp, 1])
        ofdm_symbol_im = np.reshape(np.imag(ofdm_symbol_cpx),[-1, self.nSymbol, n_sc_cp, 1])
        ofdm_symbol_real = np.concatenate([ofdm_symbol_re,ofdm_symbol_im], axis=-1)
        ofdm_pilot = ofdm_symbol_real[:,:,self.pilotCarriers,:] # Comb type pilot

        # ofdm_symbol_cpx = Clip_by_norm_np(ofdm_symbol_cpx, peak=8.0)

        return ofdm_symbol_cpx, ofdm_symbol_real, ofdm_pilot


    def ofdm_tx_frame_np(self, inputs):
        '''
        There will be no NN layers in this transmitter, only conversion
        Add frame based pilot position routation
        This will be outside the TF graph, generate data
        :param inputs: numpy array, bits, will be labels
        :param FLAGS:
        :return: OFDM data
        '''

        n_frame, frame_size, nbits = inputs.shape
        n_frame, frame_size, nbits = int(n_frame), int(frame_size), int(nbits)
        n_sc_cp = self.K+self.CP
        n_sym = n_frame * self.nSymbol
        # assert(n_sc+self.P+self.G+self.DC == self.K)
        assert(frame_size == self.frame_size)
        assert(nbits < 5)

        # Step 1, Real-2-Complex, Constellation Mapping
        const_table = const_map(nbits)
        bits = inputs.reshape([-1, nbits])
        bits.astype(np.int32)
        bits = np.pad(bits,[(0,0),(8-nbits,0)],mode='constant')
        sym_dec = np.packbits(bits, axis=1)
        # cmpx_iq = [const_table[tuple(row)] for row in bits]
        cmpx_iq = const_table.take(sym_dec) # Vectorize alternative, reduce execution time from 18 sec to <1 sec
        cmpx_iq = np.reshape(cmpx_iq, [-1, frame_size])

        # Step 2, Padding Zeros at Guard SC and DC; haven't remove DC yet
        # This code for scattered pilot pattern
        symbol = np.zeros([n_frame, self.nSymbol*self.K], dtype=np.complex64)
        symbol[:, self.dataSc] = cmpx_iq
        symbol[:, self.pilotSc] = self.pilotValue
        symbol = symbol.reshape([n_sym, self.K], order='C')
        ofdm_pilot = 3.*np.ones([n_frame, self.nSymbol, self.P, 2], dtype=np.float32)

        # Step 3: IFFT
        ofdm_time = np.fft.ifft(symbol)
        # GPU accelerated version
        # x = gpuarray.to_gpu(symbol)
        # y = gpuarray.empty(symbol.shape, np.complex64)
        # ofdm_time = fft_scikit(x, y)

        # Step 4: Add CP
        ofdm_CP = ofdm_time[:,-self.CP:]
        ofdm_symbol_cpx = np.concatenate([ofdm_CP,ofdm_time],axis=1)
        ofdm_symbol_cpx = ofdm_symbol_cpx.reshape([-1, self.nSymbol, n_sc_cp])
        ofdm_symbol_re = np.reshape(np.real(ofdm_symbol_cpx),[-1, self.nSymbol, n_sc_cp, 1])
        ofdm_symbol_im = np.reshape(np.imag(ofdm_symbol_cpx),[-1, self.nSymbol, n_sc_cp, 1])
        ofdm_symbol_real = np.concatenate([ofdm_symbol_re,ofdm_symbol_im], axis=-1)
        # ofdm_symbol_cpx = Clip_by_norm_np(ofdm_symbol_cpx, peak=8.0)

        return ofdm_symbol_cpx, ofdm_symbol_real, ofdm_pilot



def ofdm_transmitter(inputs, FLAGS):
    '''
    There will be no NN layers in this transmitter, only conversion
    :param inputs:
    :param FLAGS:
    :return:
    '''
    K = FLAGS.nfft # 64  # number of OFDM subcarriers
    CP = K // 4  # length of the cyclic prefix: 25% of the block
    P = FLAGS.npilot # 8  # number of pilot carriers per OFDM symbol
    G = FLAGS.nguard # 8  # number of guard subcarriers per OFDM symbol
    pilotValue = 3 + 3j  # The known value each pilot transmits
    guardValue = 0
    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
    effecCarriers = allCarriers[G//2:K-G//2]
    pilotCarriers = effecCarriers[::K // P]  # Pilots is every (K/P)th carrier.
    guardCarriers = np.delete(allCarriers,effecCarriers)
    # For convenience of channel estimation, let's make the last carriers also be a pilot
    # pilotCarriers = np.hstack([pilotCarriers, np.array([effecCarriers[-1]])])
    # P = P + 1

    # data carriers are all remaining carriers
    dataCarriers = np.delete(effecCarriers, pilotCarriers-G//2)

    print ("allCarriers:   %s" % effecCarriers)
    print ("pilotCarriers: %s" % pilotCarriers)
    print ("dataCarriers:  %s" % dataCarriers)

    _, n_sym, n_sc, m_iq = inputs.shape
    n_sym, n_sc, m_iq = int(n_sym), int(n_sc), int(m_iq)
    assert(n_sc+P+G == K)
    assert(m_iq < 5)

    # Step 1, Real-2-Complex, Constellation Mapping
    bits = tf.cast(tf.reshape(inputs,[-1, m_iq]), tf.int32)
    const_table = constellation_map(m_iq)
    mapping_table = tf.convert_to_tensor(const_table, tf.complex64)
    cmpx_iq = tf.gather_nd(mapping_table, bits)
    cmpx_iq = tf.reshape(cmpx_iq, [-1, n_sc])

    # Step 2, Padding Zeros at Guard SC and DC; haven't remove DC yet
    #symbol = tf.placeholder(dtype=tf.complex64, shape=(None, K))
    tf_sc = [None]*K
    for idx in range(len(dataCarriers)):
        isc = dataCarriers[idx]
        tf_sc[isc] = tf.slice(cmpx_iq, [0, idx], [-1, 1])
    for idx in range(len(pilotCarriers)):
        isc = pilotCarriers[idx]
        tf_sc[isc] = tf.multiply(tf.ones_like(tf.slice(cmpx_iq, [0, 0], [-1, 1]),dtype=tf.complex64), pilotValue)
    for idx in range(len(guardCarriers)):
        isc = guardCarriers[idx]
        tf_sc[isc] = tf.zeros_like(tf.slice(cmpx_iq, [0, 0], [-1, 1]),dtype=tf.complex64)
    symbol = tf.concat(tf_sc, axis=-1)

    # Step 3: IFFT
    ofdm_time = tf.ifft(symbol)

    # Step 4: Add CP
    n_sc_cp = K+CP
    ofdm_CP = ofdm_time[:,-CP:]
    ofdm_symbol = tf.concat([ofdm_CP,ofdm_time],axis=1)
    ofdm_symbol = tf.reshape(ofdm_symbol, [-1, n_sym, n_sc_cp])
    ofdm_symbol_real = tf.reshape(tf.real(ofdm_symbol), [-1,n_sym,n_sc_cp,1])
    ofdm_symbol_imag = tf.reshape(tf.imag(ofdm_symbol), [-1,n_sym,n_sc_cp,1])
    ofdm_symbol_re = tf.concat([ofdm_symbol_real,ofdm_symbol_imag], axis=-1)
    ofdm_pilot = ofdm_symbol_re[:,:,CP:K//P:-1,:]

    return ofdm_symbol, ofdm_symbol_re, ofdm_pilot