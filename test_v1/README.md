# Paper 
Zhongyuan Zhao, Mehmet C. Vuran, Fujuan Guo, and Stephen Scott, Deep-Waveform: A Learned OFDM Receiver Based on Deep Complex Convolutional Networks, EESS.SP, vol abs/1810.07181, Oct. 2018, [Online] https://arxiv.org/abs/1810.07181v3

[Pre-Print](https://arxiv.org/abs/1810.07181v3)

```
@article{zhao2018dcnn,
author={Zhongyuan Zhao and Mehmet C. Vuran and Fujuan Guo and Stephen Scott},
title={Deep-Waveform: A Learned OFDM Receiver Based on Deep Complex Convolutional Networks},
journal={EESS.SP},
vol = {abs/1810.07181v3},
month = {Oct},
year = {2018},
url={https://arxiv.org/abs/1810.07181v3},
}
```

**Warning**: this version of test code belongs to an [early version](https://arxiv.org/abs/1810.07181v3) of the paper. As the paper is revised substantially, this version is archieved as v1.

## Abstract
Recent explorations of Deep Learning in the physical layer (PHY) of wireless communication have shown the capabilities of Deep Neuron Networks in tasks like channel coding, modulation, and parametric estimation. However, it is unclear if Deep Neuron Networks could also learn the advanced waveforms of current and next-generation wireless networks, and potentially create new ones. In this paper, a Deep Complex Convolutional Network (DCCN) without explicit Discrete Fourier Transform (DFT) is developed as an Orthogonal Frequency-Division Multiplexing (OFDM) receiver. Compared to existing deep neuron network receivers composed of fully-connected layers followed by non-linear activations, the developed DCCN not only contains convolutional layers but is also almost (and could be fully) linear. Moreover, the developed DCCN not only learns to convert OFDM waveform with Quadrature Amplitude Modulation (QAM) into bits under noisy and Rayleigh channels, but also outperforms expert OFDM receiver based on Linear Minimum Mean Square Error channel estimator with prior channel knowledge in the low to middle Signal-to-Noise Ratios of Rayleigh channels. It shows that linear Deep Neuron Networks could learn transformations in signal processing, thus master advanced waveforms and wireless channels.

## About this code
Cross validation benchmark for Deep Learning-Based OFDM Receiver.

+ Modulation: BPSK, QPSK, 8-QAM, 16-QAM, of Gray mapping.
+ SNR: -10:1:29 dB

### Software Platform
+ Matlab 2017b
+ Tensorflow 1.10.1

### Usage
1. Run `OFDM_benchmark` in Matlab to generate .mat data for the received time-domain OFDM signal and corresponding TX bits per modulation per SNR. Each file has a size of 20MB, and total for about 3.7GB. The generated .mat data will be saved in `./mat/`
2. Run `python test_ofdm_cdnn_awgn.py --save_dir=./model/ --data_dir=./mat/` in terminal. This will load the trained models in ./model/ folder, and test the model with data stored in ./mat/ folder

### Tensors in loaded model
+ y: input bits, 
+ x: input received OFDM waveform,
+ outputs: output soft bits,
+ berlin: BER,
