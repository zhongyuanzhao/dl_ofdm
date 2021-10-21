# Paper 
Zhongyuan Zhao, Mehmet C. Vuran, Fujuan Guo, and Stephen Scott, Deep-Waveform: A Learned OFDM Receiver Based on Deep Complex-valued Convolutional Networks, in IEEE Journal on Selected Areas in Communications, vol. 39, no. 8, pp. 2407-2420, Aug. 2021, doi: [10.1109/JSAC.2021.3087241](https://doi.org/10.1109/JSAC.2021.3087241).


```
@ARTICLE{zhao2021deepwaveform,
  author={Zhao, Zhongyuan and Vuran, Mehmet Can and Guo, Fujuan and Scott, Stephen D.},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Deep-Waveform: A Learned OFDM Receiver Based on Deep Complex-Valued Convolutional Networks}, 
  year={2021},
  volume={39},
  number={8},
  pages={2407-2420},
  doi={10.1109/JSAC.2021.3087241},
}
```

[Pre-Print](https://arxiv.org/abs/1810.07181)

The majority of this work was completed during the Ph.D. studies of the author Zhongyuan Zhao at UNL. This work was supported by the NSF under Grant CNS-1731833.

UNL-CPN-Lab: [website](https://cpn.unl.edu), [github](https://github.com/UNL-CPN-Lab)

## Abstract
The (inverse) discrete Fourier transform (DFT/IDFT) is often perceived as essential to orthogonal frequency-division multiplexing (OFDM) systems. In this paper, a deep complex-valued convolutional network (DCCN) is developed to recover bits from time-domain OFDM signals without relying on any explicit DFT/IDFT. The DCCN can exploit the cyclic prefix (CP) of OFDM waveform for increased SNR by replacing DFT with a learned linear transform, and has the advantage of combining CP-exploitation, channel estimation, and intersymbol interference (ISI) mitigation, with a complexity of \mathcal{O}(N^2). Numerical tests show that the DCCN receiver can outperform the legacy channel estimators based on ideal and approximate linear minimum mean square error (LMMSE) estimation and a conventional CP-enhanced technique in Rayleigh fading channels with various delay spreads and mobility. The proposed approach benefits from the expressive nature of complex-valued neural networks, which, however, currently lack support from popular deep learning platforms. In response, guidelines of exact and approximate implementations of a complex-valued convolutional layer are provided for the design and analysis of convolutional networks for wireless PHY. Furthermore, a suite of novel training techniques are developed to improve the convergence and generalizability of the trained model in fading channels. This work demonstrates the capability of deep neural networks in processing OFDM waveforms and the results suggest that the FFT processor in OFDM receivers can be replaced by a hardware AI accelerator.

## About this code
Source code for Deep Learning-Based OFDM Receiver.

+ Modulation: BPSK, QPSK, 8-QAM, 16-QAM, of Gray mapping.
+ SNR: -10:1:29 dB

### Software platform
+ Matlab R2017b, R2018a (replace `rayleighchan` in the code for newer release)
+ Python3 compatiable
+ TensorFlow 1.x: `tensorflow-gpu==1.15`, docker tensorflow image [1.15.5-gpu-jupyter](https://hub.docker.com/layers/tensorflow/tensorflow/1.15.5-gpu-jupyter/images/sha256-5f2338b5816cd73ea82233e2dd1ee0d8e2ebf539e1e8b5741641c1e082897521?context=explore
) is highly recommended if you just want a quick tryout. 
+ **Note:** newer versions of Matlab and Tensorflow are possible, but require quite some work on the code (read [here](#for-newer-versions-of-matlab-and-tensorflow)). 

### Contents of directories
```bash
.
├── dev # latest working source code
├── test_v1 # archived for old version https://arxiv.org/abs/1810.07181v3
├── README.md 
└── LICENSE
```


### Usage
1. Run `script_rayleigh` in Matlab for benchmarks
2. Run `python3 run_local_ofdm.py --awgn=True` in terminal for training and testing results. 

### For newer versions of Matlab and Tensorflow
**Matlab**: `rayleighchan` was removed and replaced by ['comm.RayleighChannel'](https://www.mathworks.com/help/comm/ref/comm.rayleighchannel-system-object.html) from later versions of matlab. You may uncomment the lines 202-210, lines 293-300 in [/dev/m/OFDM_Benchmark_dev.m](/dev/m/OFDM_Benchmark_dev.m) to use the newer function. However, you need either replace `parfor` to `for` in that code to directly use the newer function, or change the code to initialize multiple identical objects of `comm.RayleighChannel` in lines 202-210 and use different objects (lines 293-300) in each parallel loop to enable the `parfor`. 

**Tensorflow:** I use `tf.contrib.graph_editor` to enable the transfer learning scheme described in the paper. However, in Tensorflow 2, the `tf.contrib` is removed and `graph_editor` no longer exists. If you want to use TF2 rather than TF1, you will need to re-write the transfer learning (lines 264-365 in [ofdmreceiver_np_mp.py](/dev/py//ofdmreceiver_np_mp.py)) with whatever equivalent new methods in TF 2.
