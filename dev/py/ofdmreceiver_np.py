#!/usr/bin/env python
######################################################################################
# DCCN basic receiver in tensorflow
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
import os
import time
# from sklearn.preprocessing import OneHotEncoder
from model import *
from ofdm import *
from radio import *
from util import *
# these ones let us draw images in our notebook

flags = tf.app.flags
flags.DEFINE_string('save_dir', './output/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('nbits', 1, 'bits per symbol')
flags.DEFINE_integer('msg_length', 100800, 'Message Length of Dataset')
flags.DEFINE_integer('batch_size', 512, '')
flags.DEFINE_integer('max_epoch_num', 1000, '')
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_integer('nfft', 64, 'Dropout rate TX conv block')
flags.DEFINE_integer('nsymbol', 7, 'Dropout rate TX conv block')
flags.DEFINE_integer('npilot', 8, 'Dropout rate TX dense block')
flags.DEFINE_integer('nguard', 8, 'Dropout rate RX conv block')
flags.DEFINE_integer('nfilter', 80, 'Dropout rate RX conv block')
flags.DEFINE_float('SNR', 3.0, '')
flags.DEFINE_integer('early_stop',100,'number of epoches for early stop')
flags.DEFINE_boolean('ofdm',True,'If add OFDM layer')
flags.DEFINE_string('pilot', 'lte', 'Pilot type: lte(default), block, comb, scattered')
flags.DEFINE_string('channel', 'EPA', 'AWGN or Rayleigh Channel: Flat, EPA, EVA, ETU')
flags.DEFINE_boolean('cp',True,'If include cyclic prefix')
flags.DEFINE_boolean('longcp',True,'Length of cyclic prefix: true 25%, false 7%')
flags.DEFINE_boolean('load_model',False,'Set True if run a test')
flags.DEFINE_float('split',1.0,'split factor for validation set, no split by default')
flags.DEFINE_string('token', 'OFDM','Name of model to be saved')
flags.DEFINE_boolean('test',False,'Test trained model')
FLAGS = flags.FLAGS





def test_model(FLAGS, path_prefix_min, ofdmobj, session):
    y, x, iq_receiver, outputs, total_loss, ber, berlin, conf_matrix, power_tx, noise_pwr, iq_rx, iq_tx, ce_mean, SNR = load_model_np(path_prefix_min,session)
    print("Final Test SNR: -10-30 dB")
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
        test_xs,_ = fading.run(iq_tx_cmpx)
        snr_test = snr_t * np.ones((frame_cnt, 1))
        test_xs, pwr_noise_avg = AWGN_channel_np(test_xs, snr_test)
        confmax, berl, pwr_tx, pwr_noise, test_loss, tx_sample, rx_sample = session.run([conf_matrix, berlin, power_tx, noise_pwr, ce_mean, iq_tx, iq_rx], {x: test_xs, y: test_ys, SNR:snr_test})

        print("SNR: %.2f, BER: %.8f, Loss: %f"%(snr_t, berl, test_loss))
        print("Test Confusion Matrix: ")
        print(str(confmax))
        df = df.append({'SNR': snr_t, 'BER': berl, 'Loss': test_loss}, ignore_index=True)

    df = df.set_index('SNR')
    csvfile = 'Test_DCCN_%s.csv'%(FLAGS.token + '_' + FLAGS.channel)
    df.to_csv(csvfile)

    session.close()


def main(argv):
    nbits = FLAGS.nbits # BPSK: 2, QPSK, 4, 16QAM: 16
    m_order = np.exp2(nbits)
    ofdmobj = ofdm_tx(FLAGS)

    nfft = ofdmobj.K
    ofdm_pf = FLAGS.nsymbol
    # frame_size = nfft - ofdmobj.G - ofdmobj.P - ofdmobj.DC
    frame_size = ofdmobj.frame_size
    tx_frame_size = nfft + ofdmobj.CP
    msg_length = FLAGS.msg_length
    frame_cnt = FLAGS.msg_length//FLAGS.nsymbol
    # SNR = FLAGS.SNR # set 7dB Signal to Noise Ratio
    np.random.seed(FLAGS.seed)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if FLAGS.test:
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        path_prefix_min = os.path.join(FLAGS.save_dir, FLAGS.token)
        test_model(FLAGS, path_prefix_min, ofdmobj, session)
        session.close()
        return

    tf.reset_default_graph()
    # Input x symbol stream
    y = tf.placeholder(tf.int32, shape=[None, frame_size, nbits], name='bits_in')
    x = tf.placeholder(tf.float32, shape=[None, ofdm_pf, tx_frame_size, 2], name='tx_ofdm')
    with tf.name_scope('transmitter') as scope:
        # Transmitter
        # iq_tx_re = tf.contrib.layers.layer_norm(x, center=False, scale=False, begin_norm_axis=2)/np.sqrt(2)
        batch_mean2, batch_var2 = tf.nn.moments(x, [0])
        iq_tx_re = tf.nn.batch_normalization(x, batch_mean2, batch_var2, offset=None, scale=None, variance_epsilon=1e-9)/np.sqrt(2)
        # iq_tx_re = tf.layers.batch_normalization(x, center=False, scale=False, trainable=False)/np.sqrt(2)
        iq_layer, power_tx = complex_clip(iq_tx_re, peak=8.0) # Clip by norm PAPR 8:1

    # AWGN Channel
    SNR = tf.placeholder(tf.float32, shape=[None, 1], name='SNR')  #
    with tf.name_scope('channel') as scope:
        iq_receiver, noise_pwr = AWGN_channel(iq_layer, SNR)
        # rx_iq_data = iq_receiver
        rx_iq_data = iq_tx_re  # bypass the tf AWGN

    # Receiver
    with tf.name_scope('receiver') as scope:
        # rx_layer_pulse = conv_block_rx(iq_receiver, FLAGS, filters=[4, 8], kernels=[5, 3])
        # rx_layer_symbol = dense_block_rx(rx_layer_pulse, FLAGS, outshape=[-1, frame_size, nbits, 2])
        rx_layer_symbol = ofdm_dense_rx(rx_iq_data, FLAGS, ofdmobj, outshape=[-1, frame_size, nbits, 2])
        # rx_layer_symbol = ofdm_DNN_rx(rx_iq_data, FLAGS, ofdmobj, outshape=[-1, frame_size, nbits, 2])
        outputs = rx_layer_symbol
        # print(rx_layer_pulse.shape)
        print(outputs.shape)

    print(iq_layer.shape)
    iq_tx = tf.cast(tf.reshape(iq_layer, [-1, 2]), tf.float16) # output for constellation plot
    iq_rx = tf.cast(tf.reshape(iq_receiver, [-1, 2]), tf.float16) # output for constellation plot
    # Loss Function, Confusion Matrix, Optimizer, Saver
    y_onehot = tf.one_hot(tf.reshape(y, [-1]), 2)
    out_softmax = tf.reshape(outputs, [-1, 2])
    if tf.__version__ == '1.4.0':
        crossentroy = tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=out_softmax)
    else:
        crossentroy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_onehot, logits=out_softmax)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    REG_COEFF = 0.0001
    BER_COEFF = 1.0
    ce_mean = tf.reduce_mean(crossentroy, name='ce_mean')
    y_index = tf.reshape(y, [-1])
    out_index = tf.cast(tf.argmax(out_softmax, axis=1), tf.int32)
    #BER = tf.reduce_mean(tf.abs(out_index - x_index)) # MSE
    conf_matrix = tf.confusion_matrix(y_index, out_index)
    ber, berlin = ber_tensor(conf_matrix)
    # snr_ce = tf.losses.mean_squared_error(SNR, snr_est)
    total_loss = ce_mean + berlin * REG_COEFF * sum(regularization_losses) + BER_COEFF * tf.cast(ber, tf.float32)
    tf.identity(iq_layer, 'tx_signal')
    tf.identity(power_tx, 'tx_power')
    tf.identity(rx_iq_data, 'input')
    tf.identity(outputs, 'output')
    # tf.identity(snr_est, 'snr_est')
    tf.identity(total_loss, 'cost')
    tf.identity(ber, 'log_ber')
    tf.identity(berlin, 'linear_ber')
    tf.identity(conf_matrix, 'conf_matrix')
    tf.identity(noise_pwr, 'noise_power')
    tf.identity(iq_rx, 'iq_rx')
    tf.identity(iq_tx, 'iq_tx')

    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    learning_rate = tf.train.exponential_decay(0.001, global_step_tensor,
                                               500, 0.98, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    #tf.identity(train_op,'train_op')

    saver = tf.train.Saver()

    # train for an epoch and visualize
    berl = 0.5
    batch_size = FLAGS.batch_size//ofdm_pf
    test_loss_min = 100.
    epoch_min_loss = 0
    path_prefix_min = ''
    max_epoch_num = FLAGS.max_epoch_num
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    print("Start Training")
    # snr_seq = np.array([-3.0, 0.0, 0, 0, 0, 3, 5, 5], dtype=np.float32)
    snr_seq = np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    snr_seq = snr_seq.reshape([8, 1])

    fading = rayleigh_chan_lte(FLAGS, ofdmobj.Fs)

    for epoch in range(max_epoch_num):
        print("Epoch: %d"%(epoch))
        np.random.seed(int(time.time()) + epoch)

        train_loss_avg=[]
        pwr_tx_avg=[]
        pwr_noise_avg=[]
        berl_avg=[]

        train_ys = bit_source(nbits, frame_size, frame_cnt)
        # train_snr = FLAGS.SNR * np.ones((batch_size // 8, 1))
        # train_snr = np.sqrt(FLAGS.nbits) * np.random.randn(frame_cnt, 1) + FLAGS.SNR + 6.0
        # train_snr -= max(min(float(epoch)*0.002, 6.0), 2.0)
        # train_snr[batch_size//32:] += -2.0 # 1/8 high SNR
        train_snr = FLAGS.SNR + np.repeat(snr_seq, frame_cnt//8, axis=0)
        # iq_tx_cmpx, train_xs, iq_pilot_tx = ofdmobj.ofdm_tx_np(train_ys)
        iq_tx_cmpx, train_xs, iq_pilot_tx = ofdmobj.ofdm_tx_frame_np(train_ys)
        train_xs, chan_xs = fading.run(iq_tx_cmpx)
        train_xs, pwr_noise_avg = AWGN_channel_np(train_xs, train_snr)
        for i in range(frame_cnt // batch_size):
            batch_ys = train_ys[i*batch_size:(i+1)*batch_size, :, :]
            batch_snr = train_snr[i*batch_size+np.arange(0,batch_size), :]
            batch_xs = train_xs[i*batch_size+np.arange(0,batch_size), :, :, :]
            _, pwr_tx, train_loss, berl = session.run([train_op, power_tx, ce_mean, berlin], {x: batch_xs, y:batch_ys, SNR:batch_snr})
            train_loss_avg.append(train_loss)
            pwr_tx_avg.append(pwr_tx)
            # pwr_noise_avg.append(pwr_noise)
            berl_avg.append(berl)
            # snr_mses.append(snr_mse)
        berl_mean = np.mean(berl)
        train_loss_epoch = np.mean(train_loss_avg)
        idealbatchsize = (int(min(200.0/max(berl_mean,1.e-6), 900000.)/(55*nbits))//8)
        batch_size = max(batch_size, idealbatchsize)
        print("Training Results")
        print("Tx Power: %f, Noise Power: %f" % (np.mean(pwr_tx_avg), np.mean(pwr_noise_avg)))
        # print("Tx Power: %f, Noise Power: %f, SNR MSE: %f" % (np.mean(pwr_tx_avg), np.mean(pwr_noise_avg), np.mean(snr_mses)))
        print("Train Loss: %f"%(train_loss_epoch))

        test_ys = bit_source(nbits, frame_size, 1024)
        # iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_np(test_ys)
        iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_frame_np(test_ys)
        test_xs, _ = fading.run(iq_tx_cmpx)
        snr_test = FLAGS.SNR * np.ones((1024, 1))
        test_xs, pwr_noise_avg = AWGN_channel_np(test_xs, snr_test)
        # snr_test = FLAGS.SNR + np.repeat(snr_seq, 1028//8,axis=0)
        confmax, berl, pwr_tx, test_loss, tx_sample, rx_sample = session.run([conf_matrix, berlin, power_tx, ce_mean, iq_tx, iq_rx], {x: test_xs, y: test_ys, SNR:snr_test})
        print("Test Results")
        print("Tx Power: %f, Noise Power: %f" % (pwr_tx, np.mean(pwr_noise_avg)))
        print("Test Loss: %f" % (test_loss))
        print("Test BER: %.8f" % (berl))
        print("Test Confusion Matrix: ")
        print(str(confmax))

        np.savetxt("%s_txiq.csv"%(FLAGS.token), tx_sample[0:2048], delimiter=",")
        np.savetxt("%s_rxiq.csv"%(FLAGS.token), rx_sample[0:2048], delimiter=",")

        #if (epoch%100)==0:
        if train_loss_epoch < test_loss_min:
            epoch_min_loss = epoch
            test_loss_min = train_loss_epoch
            # path_prefix_min = saver.save(session, os.path.join(FLAGS.save_dir, FLAGS.token), global_step=global_step_tensor)
            path_prefix_min = saver.save(session, os.path.join(FLAGS.save_dir, FLAGS.token))
        if epoch - FLAGS.early_stop > epoch_min_loss:
            break

    # path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, FLAGS.token))
    print("Training Done!, Best model saved to")
    print(path_prefix_min)

    #sess = tf.Session()

    test_model(FLAGS, path_prefix_min, ofdmobj, session)
    session.close()



if __name__ == "__main__":
    tf.app.run()

