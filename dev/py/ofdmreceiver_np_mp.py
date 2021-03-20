#!/usr/bin/env python
######################################################################################
# DCCN equalized receiver in tensorflow
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
import pandas as pd
import scipy.io as sio
import os
import time
# from sklearn.preprocessing import OneHotEncoder
from model import *
from ofdm import *
from radio import *
from util import *
import copy
# these ones let us draw images in our notebook

flags = tf.app.flags
flags.DEFINE_string('save_dir', './output/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('nbits', 1, 'bits per symbol')
flags.DEFINE_integer('msg_length', 100800, 'Message Length of Dataset')
flags.DEFINE_integer('batch_size', 512, '')
flags.DEFINE_integer('max_epoch_num', 5000, '')
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_integer('nfft', 64, 'Dropout rate TX conv block')
flags.DEFINE_integer('nsymbol', 7, 'Dropout rate TX conv block')
flags.DEFINE_integer('npilot', 8, 'Dropout rate TX dense block')
flags.DEFINE_integer('nguard', 8, 'Dropout rate RX conv block')
flags.DEFINE_integer('nfilter', 80, 'Dropout rate RX conv block')
flags.DEFINE_float('SNR', 30.0, '')
flags.DEFINE_float('SNR2', 30.0, '')
flags.DEFINE_integer('early_stop',400,'number of epoches for early stop')
flags.DEFINE_boolean('ofdm',True,'If add OFDM layer')
flags.DEFINE_string('pilot', 'lte', 'Pilot type: lte(default), block, comb, scattered')
flags.DEFINE_string('channel', 'EPA', 'AWGN or Rayleigh Channel: Flat, EPA, EVA, ETU')
flags.DEFINE_boolean('cp',True,'If include cyclic prefix')
flags.DEFINE_boolean('longcp',True,'Length of cyclic prefix: true 25%, false 7%')
flags.DEFINE_boolean('load_model',True,'Set True if run a test')
flags.DEFINE_float('split',1.0,'split factor for validation set, no split by default')
flags.DEFINE_string('token', 'OFDM','Name of model to be saved')
flags.DEFINE_integer('opt', 3, '0: default equalizer, 1: NoCConv, 2: NoResidual, 3: DNN')
flags.DEFINE_boolean('mobile', False, 'If Doppler spread is turned on')
flags.DEFINE_float('init_learning', 0.001, '')
flags.DEFINE_boolean('test',False,'Test trained model')
FLAGS = flags.FLAGS


def test_model_cross(FLAGS, path_prefix_min, ofdmobj, session):
    y, x, iq_receiver, outputs, total_loss, ber, berlin, conf_matrix, power_tx, noise_pwr, iq_rx, iq_tx, ce_mean, SNR = load_model_np(path_prefix_min,session)
    print("Final Test SNR: -10 to 30 dB")
    nfft = FLAGS.nfft
    nbits = FLAGS.nbits
    npilot = FLAGS.npilot # last carrier as pilot
    nguard = FLAGS.nguard
    nsymbol = FLAGS.nsymbol
    DC = 2
    np.random.seed(int(time.time()))
    frame_size = ofdmobj.frame_size
    frame_cnt = 30000
    for test_chan in ['ETU','EVA','EPA','Flat', 'Custom']:
        df = pd.DataFrame(columns=['SNR', 'BER', 'Loss'])
        flagcp = copy.deepcopy(FLAGS)
        flagcp.channel = test_chan
        # fading = rayleigh_chan_lte(flagcp, ofdmobj.Fs, mobile=FLAGS.mobile)
        fading = RayleighChanParallel(flagcp, ofdmobj.Fs, mobile=FLAGS.mobile)
        print("Test in %s, mobile: %s"%(test_chan, FLAGS.mobile))
        for snr_t in range(-10, 31, 5):
            np.random.seed(int(time.time()) + snr_t)
            test_ys = bit_source(nbits, frame_size, frame_cnt)
            # iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_np(test_ys)
            iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_frame_np(test_ys)
            test_xs, _ = fading.run(iq_tx_cmpx)
            snr_test = snr_t * np.ones((frame_cnt, 1))
            test_xs, pwr_noise_avg = AWGN_channel_np(test_xs, snr_test)
            confmax, berl, pwr_tx, pwr_noise, test_loss, tx_sample, rx_sample = session.run([conf_matrix, berlin, power_tx, noise_pwr, ce_mean, iq_tx, iq_rx], {x: test_xs, y: test_ys, SNR:snr_test})

            print("SNR: %.2f, BER: %.8f, Loss: %f"%(snr_t, berl, test_loss))
            print("Test Confusion Matrix: ")
            print(str(confmax))
            df = df.append({'SNR': snr_t, 'BER': berl, 'Loss': test_loss}, ignore_index=True)

        df = df.set_index('SNR')
        # csvfile = 'Test_DCCN_%s_test_chan_%s.csv'%(FLAGS.token + '_Equalizer_' + FLAGS.channel, test_chan)
        if FLAGS.mobile:
            csvfile = 'Test_DCCN_%s_test_chan_%s_mobile.csv' % (FLAGS.token + '_Equalizer%d_' % (FLAGS.opt) + FLAGS.channel, test_chan)
        else:
            csvfile = 'Test_DCCN_%s_test_chan_%s.csv'%(FLAGS.token + '_Equalizer%d_'%(FLAGS.opt) + FLAGS.channel, test_chan)
        df.to_csv(csvfile)

    session.close()



def test_model(FLAGS, path_prefix_min,ofdmobj,session):
    y, x, iq_receiver, outputs, total_loss, ber, berlin, conf_matrix, power_tx, noise_pwr, iq_rx, iq_tx, ce_mean, SNR = load_model_np(path_prefix_min,session)
    print("Final Test SNR: -10 to 30 dB")
    nfft = FLAGS.nfft
    nbits = FLAGS.nbits
    npilot = FLAGS.npilot # last carrier as pilot
    nguard = FLAGS.nguard
    nsymbol = FLAGS.nsymbol
    DC = 2
    frame_size = ofdmobj.frame_size
    frame_cnt = 20000
    df = pd.DataFrame(columns=['SNR','BER','Loss'])
    fading = rayleigh_chan_lte(FLAGS, ofdmobj.Fs)
    for snr_t in range(-10, 31):
        np.random.seed(int(time.time()) + snr_t)
        test_ys = bit_source(nbits, frame_size, frame_cnt)
        # iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_np(test_ys)
        iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_frame_np(test_ys)
        test_xs, _ = fading.run(iq_tx_cmpx)
        snr_test = snr_t * np.ones((frame_cnt, 1))
        test_xs, pwr_noise_avg = AWGN_channel_np(test_xs, snr_test)
        confmax, berl, pwr_tx, pwr_noise, test_loss, tx_sample, rx_sample = session.run([conf_matrix, berlin, power_tx, noise_pwr, ce_mean, iq_tx, iq_rx], {x: test_xs, y: test_ys, SNR:snr_test})

        print("SNR: %.2f, BER: %.8f, Loss: %f"%(snr_t, berl, test_loss))
        print("Test Confusion Matrix: ")
        print(str(confmax))
        df = df.append({'SNR': snr_t, 'BER': berl, 'Loss': test_loss}, ignore_index=True)

    df = df.set_index('SNR')
    csvfile = 'Test_DCCN_%s.csv'%(FLAGS.token + '_Equalizer_' + FLAGS.channel)
    df.to_csv(csvfile)

    session.close()


def test_model_mat(FLAGS, path_prefix_min,ofdmobj,session):
    y, x, iq_receiver, outputs, total_loss, ber, berlin, conf_matrix, power_tx, noise_pwr, iq_rx, iq_tx, ce_mean, SNR = load_model_np(path_prefix_min,session)
    print("Final Test SNR: -10 to 30 dB")
    mod_names = ['BPSK','QPSK','8QAM','16QAM']
    data_dir = '../m/mat'
    nfft = ofdmobj.K
    nbits = FLAGS.nbits
    nsymbol = FLAGS.nsymbol
    n_sc = ofdmobj.CP + ofdmobj.K
    frame_size = ofdmobj.frame_size
    frame_cnt = 20000
    msg_length = frame_cnt*nsymbol
    if FLAGS.longcp:
        cpstr = ''
    else:
        cpstr = '_shortcp'

    df = pd.DataFrame(columns=['SNR','BER','Loss'])

    # load matlab data
    mat_name = 'TX_bit_iq_%s_%s_FFT%d%s.mat'%(mod_names[nbits-1], FLAGS.channel, nfft, cpstr)
    matfile = sio.loadmat(os.path.join(data_dir, mat_name))
    iq_matlab = matfile['Ch_Data']
    txbits_matlab = matfile['txbits']
    test_xs = np.transpose(iq_matlab, axes=[1,0])
    test_xs = np.reshape(test_xs, [frame_cnt, nsymbol, -1])
    xs_real = np.reshape(np.real(test_xs), [frame_cnt, nsymbol, n_sc, 1])
    xs_imag = np.reshape(np.imag(test_xs), [frame_cnt, nsymbol, n_sc, 1])
    chan_xs = 3*np.concatenate([xs_real, xs_imag], axis=-1)
    test_ys = np.reshape(txbits_matlab, [frame_cnt, frame_size, nbits])

    for snr_t in range(-10, 31):
        snr_test = snr_t * np.ones((frame_cnt, 1))
        test_xs, pwr_noise_avg = AWGN_channel_np(chan_xs, snr_test)
        confmax, berl, pwr_tx, pwr_noise, test_loss, tx_sample, rx_sample = session.run([conf_matrix, berlin, power_tx, noise_pwr, ce_mean, iq_tx, iq_rx], {x: test_xs, y: test_ys, SNR:snr_test})

        print("SNR: %.2f, BER: %.8f, Loss: %f"%(snr_t, berl, test_loss))
        print("Test Confusion Matrix: ")
        print(str(confmax))
        df = df.append({'SNR': snr_t, 'BER': berl, 'Loss': test_loss}, ignore_index=True)

    df = df.set_index('SNR')
    csvfile = 'Test_DCCN_%s.csv'%(FLAGS.token + '_Equalizer_' + FLAGS.channel)
    df.to_csv(csvfile)

    session.close()


class RayleighChanParallel:
    def __init__(self, flags, sample_rate=0.96e6, mobile=False, mix=False):
        self.cpu_count = mp.cpu_count()
        self.Fs = sample_rate
        self.flags = flags
        self.mobile = mobile
        self.mix = mix
        self.nfft = flags.nfft
        self.pool = mp.Pool(processes=self.cpu_count)
        self.create_objs()

    def create_objs(self):
        objs = []
        for i in range(self.cpu_count):
            fading_obj = rayleigh_chan_lte(self.flags, self.Fs, self.mobile, self.mix)
            objs.append(fading_obj)
        self.objs = objs

    def run(self, iq_tx_cmpx):
        n_fr, n_sym, n_sc = np.shape(iq_tx_cmpx)
        tx_signal_list = []
        chunk_size = np.ceil(n_fr/self.cpu_count).astype(int)
        for i in range(self.cpu_count):
            tx_chuck = iq_tx_cmpx[i*chunk_size:(i+1)*chunk_size, :, :]
            tx_signal_list.append(tx_chuck)

        results = [self.pool.apply(self.objs[i], args=(tx_signal_list[i],)) for i in range(self.cpu_count)]
        rx_signal = np.zeros([n_fr, n_sym, n_sc, 2], dtype=np.float)
        ch_ground = np.zeros([n_fr, n_sym, self.nfft], dtype=np.complex64)
        for i in range(self.cpu_count):
            rx_sig_sym, chan_sym = results[i]
            rx_signal[i*chunk_size:(i+1)*chunk_size, :, :, :] = rx_sig_sym
            ch_ground[i*chunk_size:(i+1)*chunk_size, :, :] = chan_sym
        return rx_signal, ch_ground



def main(argv):
    nbits = FLAGS.nbits # BPSK: 2, QPSK, 4, 16QAM: 16
    m_order = np.exp2(nbits)
    ofdmobj = ofdm_tx(FLAGS)

    nfft = ofdmobj.K
    ofdm_pf = ofdmobj.nSymbol # 8
    # frame_size = nfft - ofdmobj.G - ofdmobj.P - ofdmobj.DC
    frame_size = ofdmobj.frame_size
    msg_length = FLAGS.msg_length
    frame_cnt = FLAGS.msg_length//FLAGS.nsymbol
    #SNR = FLAGS.SNR # set 7dB Signal to Noise Ratio
    np.random.seed(FLAGS.seed)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    layer_norm = tf.keras.layers.LayerNormalization(axis=1, center=False, scale=False)

    if FLAGS.test:
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        # if FLAGS.opt == 0:
        #     path_prefix_min = os.path.join(FLAGS.save_dir, FLAGS.token + '_Equalizer_' + FLAGS.channel)
        # else:
        #     path_prefix_min = os.path.join(FLAGS.save_dir, FLAGS.token + '_Equalizer%d_'%(FLAGS.opt)  + FLAGS.channel)
        path_prefix_min = os.path.join(FLAGS.save_dir, FLAGS.token + '_Equalizer%d_'%(FLAGS.opt)  + FLAGS.channel)
        test_model_cross(FLAGS, path_prefix_min, ofdmobj, session)
        session.close()
        return

    # if FLAGS.load_model:
    #
    # else:
    #     pass

    session = tf.Session(config=config)
    saver = tf.train.import_meta_graph(os.path.join(FLAGS.save_dir, FLAGS.token + '.meta'))
    saver.restore(session, FLAGS.save_dir + FLAGS.token)
    # print(session.graph.get_operations())
    graph = session.graph
    sgv_tx = tf.contrib.graph_editor.sgv_scope('transmitter', graph=graph)
    sgv_rx = tf.contrib.graph_editor.sgv_scope('receiver', graph=graph)
    sgv_ch_awgn = tf.contrib.graph_editor.sgv_scope('channel', graph=graph)

    y = graph.get_tensor_by_name('bits_in:0')
    x = graph.get_tensor_by_name('tx_ofdm:0')
    SNR = graph.get_tensor_by_name('SNR:0')
    iq_layer = graph.get_tensor_by_name('tx_signal:0')
    power_tx = graph.get_tensor_by_name('tx_power:0')
    iq_tx = graph.get_tensor_by_name('iq_tx:0')
    iq_rx = graph.get_tensor_by_name('iq_rx:0')
    rx_iq_data = graph.get_tensor_by_name('input:0')
    ce_mean = graph.get_tensor_by_name('ce_mean:0')
    noise_pwr = graph.get_tensor_by_name('noise_power:0')
    ber = graph.get_tensor_by_name('log_ber:0')
    berlin = graph.get_tensor_by_name('linear_ber:0')
    conf_matrix = graph.get_tensor_by_name('conf_matrix:0')
    # snr_est = graph.get_tensor_by_name('snr_est:0')
    # out_fft = graph.get_tensor_by_name('receiver/fft_like/fft_out:0')
    # session.close()
    chan_gt = tf.placeholder(tf.complex64, shape=[None, ofdm_pf, nfft], name='chan_freq')

    iq_rx_mp = tf.cast(tf.reshape(rx_iq_data, [-1, 2]), tf.float16) # output for constellation plot
    with tf.variable_scope('Equalizer') as scope:
        if FLAGS.opt == 0:
            out_eq, snr_est, chest = equalizer_ofdm(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 1:
            out_eq, snr_est, chest = equalizer_nocconv(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 2:
            out_eq, snr_est, chest = equalizer_noresdl(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 4:
            out_eq, snr_est, chest = equalizer_noresdl2(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 5:
            out_eq, snr_est, chest = equalizer_noresdl4(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 3:
            out_eq, snr_est, chest = equalizer_dnnE(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 6:
            out_eq, snr_est, chest = equalizer_doppler(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 7:
            out_eq, snr_est, chest = equalizer_separateIQ(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 9:
            out_eq, snr_est, chest = equalizer_ofdm(rx_iq_data, FLAGS, ofdmobj)
        elif FLAGS.opt == 10:
            out_eq, snr_est, chest = equalizer_ofdm(rx_iq_data, FLAGS, ofdmobj)
        # out_eq, snr_est = equalizer_freq(out_fft, FLAGS)
    with tf.variable_scope('dummy') as scope:
        iq_rx_eq = out_eq + 0
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Equalizer')
    sgv_rx_dummy = tf.contrib.graph_editor.sgv_scope('dummy', graph=graph)
    # tf.contrib.graph_editor.reroute_outputs(sgv_ch_awgn, sgv_eq)
    sgv_rx_new = sgv_rx.remap_inputs([0])
    tf.contrib.graph_editor.reroute_inputs(sgv_rx_dummy, sgv_rx_new)
    # sgv_rx_demod1 = sgv_rx_demod.remap_inputs([0])
    # tf.contrib.graph_editor.reroute_inputs(sgv_rx_dummy, sgv_rx_demod1)

    snr_ce = tf.losses.mean_squared_error(SNR, snr_est)
    chan_src = tf.reshape(chan_gt, [-1, ofdm_pf, nfft, 1])
    chan_src = tf.concat([tf.math.real(chan_src), tf.math.imag(chan_src)], axis=-1)
    # norm_chan_gt = tf.contrib.layers.layer_norm(chan_src, center=False, scale=False, begin_norm_axis=1)
    norm_chan_gt = layer_norm(chan_src)
    chest = tf.reshape(chest, [-1, ofdm_pf, nfft, 1])
    chest = tf.concat([tf.math.real(chest), tf.math.imag(chest)], axis=-1)
    # norm_chest = tf.contrib.layers.layer_norm(chest, center=False, scale=False, begin_norm_axis=1)
    norm_chest = layer_norm(chest)
    chan_rms = tf.losses.mean_squared_error(norm_chan_gt, norm_chest)

    with tf.variable_scope('optimizer') as scope:
        # equalization_losses = tf.losses.mean_squared_error(pilot_eq, iq_pilot_tx)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        REG_COEFF = 0.001
        BER_COEFF = 1.0
        total_loss = ce_mean \
                     + REG_COEFF * sum(regularization_losses) \

        global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        learning_rate = tf.train.exponential_decay(FLAGS.init_learning, global_step_tensor,
                                                   500, 0.98, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, var_list=trainable_vars, global_step=global_step_tensor)

    if FLAGS.opt == 0:
        save_model_name = FLAGS.token + '_Equalizer_' + FLAGS.channel
    else:
        save_model_name = FLAGS.token + '_Equalizer%d_' % (FLAGS.opt) + FLAGS.channel

    saver = tf.train.Saver()

    # train for an epoch and visualize
    berl = 0.5
    batch_size = FLAGS.batch_size//ofdm_pf
    test_loss_min = 100.
    epoch_min_loss = 0
    path_prefix_min = ''
    max_epoch_num = FLAGS.max_epoch_num
    if not FLAGS.load_model:
        session = tf.Session()
        session.run(tf.global_variables_initializer())
    else:
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Equalizer')
        trainable_variable_initializers = [var.initializer for var in trainable_vars]
        session.run(trainable_variable_initializers)
        optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "optimizer")
        session.run(tf.variables_initializer(optimizer_vars, name='init'))
        saver.save(session, os.path.join(FLAGS.save_dir, FLAGS.token+'edit'))
        session = tf.Session()
        # load edited graph
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.save_dir, FLAGS.token + 'edit.meta'))
        saver.restore(session, FLAGS.save_dir + FLAGS.token+'edit')
        # load trained graph
        # saver = tf.train.import_meta_graph(os.path.join(FLAGS.save_dir, save_model_name + '.meta'))
        # saver.restore(session, FLAGS.save_dir + save_model_name)


    print("Start Training")
    snr_seq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    snr_seq = snr_seq.reshape([8, 1])
    np.random.seed(int(time.time()))
    aa_milne_arr = np.linspace(0, 27, 10, dtype=np.float32)

    # fading = rayleigh_chan_lte(FLAGS, ofdmobj.Fs, FLAGS.mobile)
    fading0 = RayleighChanParallel(FLAGS, ofdmobj.Fs, mobile=False)
    if FLAGS.mobile:
        fading1 = RayleighChanParallel(FLAGS, ofdmobj.Fs, mobile=FLAGS.mobile, mix=True)

    phase2 = True
    for epoch in range(max_epoch_num):
        print("Epoch: %d"%(epoch))
        np.random.seed(int(time.time()) + epoch)

        train_loss_avg=[]
        pwr_tx_avg=[]
        pwr_noise_avg=[]
        berl_avg=[]
        snr_mses = []

        train_ys = bit_source(nbits, frame_size, frame_cnt)
        train_snr = np.random.choice(aa_milne_arr, [frame_cnt, 1], p=[0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.1, 0.5, 0.2, 0.1])
        iq_tx_cmpx, train_xs, iq_pilot_tx = ofdmobj.ofdm_tx_frame_np(train_ys)
        # train_xs, chan_xs = fading.run(iq_tx_cmpx)
        if phase2 and FLAGS.mobile:
            train_xs, chan_xs = fading1.run(iq_tx_cmpx)
        else:
            train_xs, chan_xs = fading0.run(iq_tx_cmpx)
        train_xs, pwr_noise_avg = AWGN_channel_np(train_xs, train_snr)

        for i in range(frame_cnt // batch_size):
            batch_ys = train_ys[i*batch_size:(i+1)*batch_size, :, :]
            batch_snr = train_snr[i*batch_size+np.arange(0,batch_size), :]
            batch_xs = train_xs[i*batch_size+np.arange(0,batch_size), :, :,:]
            batch_chan = chan_xs[i*batch_size+np.arange(0,batch_size), :, :]
            _, pwr_tx, train_loss, berl, snr_mse = session.run([train_op, power_tx, ce_mean, berlin, chan_rms], {x: batch_xs, y:batch_ys, SNR:batch_snr, chan_gt: batch_chan})
            train_loss_avg.append(train_loss)
            pwr_tx_avg.append(pwr_tx)
            # pwr_noise_avg.append(pwr_noise)
            berl_avg.append(berl)
            snr_mses.append(snr_mse)
        berl_mean = np.mean(berl)
        train_loss_epoch = np.mean(train_loss_avg)
        print("Training Results")
        print("Tx Power: %f, Noise Power: %f, SNR MSE: %f" % (np.mean(pwr_tx_avg), np.mean(pwr_noise_avg), np.mean(snr_mses)))
        print("Train Loss: %f"%(train_loss_epoch))

        # Test Model
        test_ys = bit_source(nbits, frame_size, 1024)
        # iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_np(test_ys)
        iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_frame_np(test_ys)
        # test_xs, chan_xs = fading.run(iq_tx_cmpx)
        if phase2 and FLAGS.mobile:
            test_xs, chan_xs = fading1.run(iq_tx_cmpx)
        else:
            test_xs, chan_xs = fading0.run(iq_tx_cmpx)
        # snr_test = FLAGS.SNR2 * np.ones((1024, 1))
        # snr_test = 20.0 * np.ones((1024, 1))
        snr_test = np.random.choice(aa_milne_arr, [1024, 1], p=[0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.1, 0.5, 0.2, 0.1])
        test_xs, pwr_noise_avg = AWGN_channel_np(test_xs, snr_test)
        # snr_test = FLAGS.SNR + np.repeat(snr_seq, 1028//8,axis=0)
        confmax, berl, pwr_tx, test_loss, tx_sample, rx_sample = session.run([conf_matrix, berlin, power_tx, ce_mean, iq_tx, iq_rx], {x: test_xs, y: test_ys, SNR:snr_test})
        print("Test Results")
        print("Tx Power: %f, Noise Power: %f" % (pwr_tx, np.mean(pwr_noise_avg)))
        print("Test Loss: %f" % (test_loss))
        print("Test BER: %.8f" % (berl))
        print("Test Confusion Matrix: ")
        print(str(confmax))

        np.savetxt("%s_mp_txiq.csv"%(FLAGS.token), tx_sample[0:2048], delimiter=",")
        np.savetxt("%s_mp_rxiq.csv"%(FLAGS.token), rx_sample[0:2048], delimiter=",")

        if train_loss_epoch < test_loss_min:
            epoch_min_loss = epoch
            test_loss_min = train_loss_epoch
            path_prefix_min = saver.save(session, os.path.join(FLAGS.save_dir, save_model_name))
        if epoch - FLAGS.early_stop > epoch_min_loss:
            if phase2 or not FLAGS.mobile:
                break
            else:
                epoch_min_loss = epoch
                test_loss_min = 1.0
                phase2 = True

    print("Training Done!, Best model saved to")
    print(path_prefix_min)

    # final test
    test_model_cross(FLAGS, path_prefix_min, ofdmobj, session)
    session.close()



if __name__ == "__main__":
    tf.app.run()

