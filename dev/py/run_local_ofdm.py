#! /usr/bin/python
######################################################################################
# Wrapper of training DCCN basic receiver and channel equalizer
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

import os
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import time
import subprocess
from locals import *

flags = tf.app.flags
flags.DEFINE_boolean('awgn',True,'set False to skip awgn model training')
FLAGS = flags.FLAGS


def main(argv):
    # Get current working directory
    cwd = os.getcwd()
    py_basic = "ofdmreceiver_np.py"
    py_equalizer = "ofdmreceiver_np_mp.py"
    token = "OFDM_Dense3"
    batchsize = 512
    learning = 0.001
    # longcp = 'False'
    longcp = 'True'
    cp = 'True'
    mobile = 'True'
    if mobile == 'True':
        mobile_str = '_mobile'
    else:
        mobile_str = ''
    nFFT = 64

    if longcp == 'True':
        save_dir = "./ofdm_lte_ext_%s_longcp%s/"%(nFFT, mobile_str)
        result_dir = './test_ext_%s_long_cross%s'%(nFFT, mobile_str)
    else:
        save_dir = "./ofdm_lte_ext_%s_shortcp%s/"%(nFFT, mobile_str)
        result_dir = './test_ext_%s_short_cross%s'%(nFFT, mobile_str)

    for folder in [save_dir, result_dir]:
        if not os.path.isdir(folder):
            os.makedirs(folder)
            print("created folder : ", folder)


    for longcp in ['False', 'True']:
        n_filter = nFFT

        nbits_set = [1] # [1,2,4]
        # nbits_set = range(1,2)
        ebno = 5.0
        if FLAGS.awgn:
            for nbits in reversed([4,3,2,1]):
                snr = float(ebno*nbits)
                max_epoch_num = 1200 * nbits
                cond = "%dmod"%(nbits)
                for cp in ['False', 'True']:
                    chan = 'AWGN'
                    parameters_i = "--channel=%s --save_dir=%s --early_stop=200 --nfilter=%d --batch_size=%d --max_epoch_num=%d --cp=%s --nfft=%d --longcp=%s " \
                                   % (chan, save_dir, n_filter, batchsize, max_epoch_num, cp, nFFT, longcp)

                    token1 = "%s_%s_snr%d_cp%s" % (token, cond, int(snr), cp)
                    parameters = parameters_i + "--SNR=%.2f --nbits=%d --token=%s"%(snr, nbits, token1)

                    csvfile = "Test_DCCN_%s_%s.csv"%(token1, chan)
                    csvdest = os.path.join(result_dir, csvfile)
                    if os.path.isfile(csvfile):
                        subprocess.call("mv %s %s" % (csvfile, result_dir), shell=True)
                        continue
                    elif os.path.isfile(csvdest):
                        continue
                    runlocalpython(parameters, py_basic)
                    time.sleep(30)

                    subprocess.call("mv %s %s"%(csvfile, result_dir), shell=True)

        nbits = 1
        # for nbits in reversed(nbits_set):
        for opt in [0]:
            snr = float(ebno * nbits)
            max_epoch_num = 4000 * nbits
            cond = "%dmod" % (nbits)
            for cp in ['True', 'False']:
                # for chan in ['EPA', 'EVA', 'Flat', 'ETU']:
                for chan in ['mixRayleigh']:
                    parameters_i = "--channel=%s --save_dir=%s --init_learning=%.4f --early_stop=200 --nfilter=%d --batch_size=%d --max_epoch_num=%d --cp=%s --nfft=%d --longcp=%s --opt=%d --mobile=%s " \
                                   % (chan, save_dir, learning, n_filter, batchsize, max_epoch_num, cp, nFFT, longcp, opt, mobile)

                    token1 = "%s_%s_snr%d_cp%s" % (token, cond, int(snr), cp)
                    parameters = parameters_i + "--SNR=%.2f --nbits=%d --token=%s"%(snr, nbits, token1)

                    csvfile = "Test_DCCN_%s_Equalizer%d_%s_test_chan_Custom.csv"%(token1, opt, chan)
                    csvfile_s = "Test_DCCN_%s_Equalizer%d_%s*.csv"%(token1, opt, chan)
                    csvdest = os.path.join(result_dir, csvfile)
                    if os.path.isfile(csvfile):
                        subprocess.call("mv %s %s" % (csvfile, result_dir), shell=True)
                        continue
                    elif os.path.isfile(csvdest):
                        continue
                    runlocalpython(parameters, py_equalizer)
                    time.sleep(30)

                    subprocess.call("mv %s %s"%(csvfile_s, result_dir), shell=True)


if __name__ == "__main__":
    tf.app.run()

