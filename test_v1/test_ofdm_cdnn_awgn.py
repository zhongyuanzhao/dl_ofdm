#! /usr/bin/python
from numpy import genfromtxt
import tensorflow as tf
import numpy as np
import os
import time
import scipy.io as sio

flags = tf.app.flags
flags.DEFINE_string('data_dir', '../m/mat/', 'directory .mat data is saved')
flags.DEFINE_string('save_dir', './model/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('nbits', 1, 'bits per symbol')
flags.DEFINE_integer('msg_length', 102400, 'Message Length of Dataset')
flags.DEFINE_integer('batch_size', 512, '')
flags.DEFINE_integer('max_epoch_num', 1000, '')
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_integer('nfft', 64, 'Dropout rate TX conv block')
flags.DEFINE_integer('nsymbol', 8, 'number of OFDM symbols per frame')
flags.DEFINE_integer('npilot', 8, 'number of pilot cells per OFDM symbol')
flags.DEFINE_integer('nguard', 8, 'number of guard bands per OFDM symbol (without DC)')
flags.DEFINE_integer('nfilter', 80, 'number of filters')
flags.DEFINE_float('SNR', 3.0, 'Signal to Noise Ratio')
flags.DEFINE_integer('early_stop',100,'number of epoches for early stop')
flags.DEFINE_string('channel', 'AWGN', 'AWGN or Rayleigh Channel')
flags.DEFINE_boolean('cp',False,'If include cyclic prefix')
flags.DEFINE_boolean('load_model',False,'Set True if run a test')
flags.DEFINE_string('token', 'OFDM','Name of model to be saved')
FLAGS = flags.FLAGS


class testflag:
    def __init__(self):
        self.token='OFDM'
        self.split=1.0
        self.load_model=True
        self.cp = False
        self.early_stop = 100
        self.SNR = 3.0
        self.nfilter=80
        self.nguard = 8
        self.npilot=8
        self.nsymbol=8
        self.nfft=64
        self.seed=1
        self.max_epoch_num=1000
        self.batch_size=512
        self.msg_length=102400
        self.nbits=1
        self.save_dir='./model/'


class ofdm_tx:
    def __init__(self, sflag):
        self.K=sflag.nfft
        self.CP = sflag.nfft//4
        self.DC = 2
        self.G = sflag.nguard
        self.P = sflag.npilot
        self.nSymbol = sflag.nsymbol
        self.nfft=64
        self.nbits=sflag.nbits


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


def test_model(path_prefix_min, ofdmobj, session):
    y, x, iq_receiver, outputs, total_loss, ber, berlin, conf_matrix, power_tx, noise_pwr, iq_rx, iq_tx, ce_mean, SNR = load_model_np(path_prefix_min,session)
    print("Final Test SNR: -10-30 dB")
    mod_names = ['BPSK','QPSK','8QAM','16QAM']
    nfft = ofdmobj.K
    nbits = ofdmobj.nbits
    npilot = ofdmobj.P # last carrier as pilot
    nguard = ofdmobj.G
    nsymbol = ofdmobj.nSymbol
    DC = ofdmobj.DC
    frame_size = nfft - nguard - npilot - DC
    msg_length = 16000
    n_fr = msg_length//nsymbol
    n_sc = nfft + ofdmobj.CP
    for snr_t in range(-10, 30):
        mat_name = '%sofdm_awgn_%s_%ddB.mat'%(FLAGS.data_dir,mod_names[nbits-1], snr_t)
        matfile = sio.loadmat(mat_name)
        iq_matlab = matfile['y']
        txbits_matlab = matfile['txbits']
        test_xs = np.transpose(iq_matlab, axes=[1,0])
        test_xs = np.reshape(test_xs,[msg_length//nsymbol,nsymbol,-1])
        xs_real = np.reshape(np.real(test_xs), [n_fr, nsymbol, n_sc, 1])
        xs_imag = np.reshape(np.imag(test_xs), [n_fr, nsymbol, n_sc, 1])
        test_xs = np.concatenate([xs_real, xs_imag], axis=-1)
        test_ys = np.reshape(txbits_matlab,[msg_length,frame_size,nbits])
        snr_test = 100 * np.ones((msg_length//nsymbol, 1)) # noise is already included in test_xs from matlab
        confmax, berl, pwr_tx, pwr_noise, test_loss, tx_sample, rx_sample = session.run([conf_matrix, berlin, power_tx, noise_pwr, ce_mean, iq_tx, iq_rx], {x: test_xs, y: test_ys, SNR:snr_test})

        print("SNR: %.2f, BER: %.8f, Loss: %f"%(snr_t, berl, test_loss))
        print("Test Confusion Matrix: ")
        print(str(confmax))
    session.close()


def main(argv):
    # Get current working directory
    cwd = os.getcwd()
    token = "OFDM_Dense3"
    # From Previous Test, max_epoch_num = 120~200 is ok
    batchsize = 512
    n_filter = FLAGS.nfilter # 128

    # save_dir = "./ofdm_saved/"
    # save_dir = "./ofdm_np_dc/"
    save_dir = FLAGS.save_dir
    data_dir = FLAGS.data_dir
    ebno = 3.0
    for nbits in range(1,5):
        snr = float(ebno*nbits)
        max_epoch_num = 1200 * nbits
        cond = "%dmod"%(nbits)
        for cp in ['False', 'True']:
            tflag = testflag()
            chan = 'AWGN'
            # chan = 'EPA'
            tflag.batch_size=batchsize
            tflag.cp=cp
            tflag.channe=chan
            tflag.save_dir=save_dir
            tflag.early_stop=100
            tflag.nfilter=n_filter
            tflag.max_epoch_num=max_epoch_num
            tflag.SNR=snr
            tflag.nbits=nbits

            token1 = "%s_%s_snr%d_cp%s" % (token, cond, int(snr), cp)
            tflag.token = token1

            path_prefix_min = '%s%s'%(FLAGS.save_dir,token1)
            ofdmobj = ofdm_tx(tflag)
            session = tf.Session()
            session.run(tf.global_variables_initializer())
            test_model(path_prefix_min,ofdmobj,session)
            session.close()
            tf.reset_default_graph()
            time.sleep(2)



if __name__ == "__main__":
    tf.app.run()

