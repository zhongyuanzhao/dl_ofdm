%% Visualize the QAM constellation mapping
% Author: Zhongyuan Zhao
% Date: 2021-03-10
% Link: https://github.com/zhongyuanzhao/dl_ofdm
% Cite this work:
% Zhongyuan Zhao, Mehmet C. Vuran, Fujuan Guo, and Stephen Scott, "Deep-Waveform: 
% A Learned OFDM Receiver Based on Deep Complex-valued Convolutional Networks," 
% EESS.SP, vol abs/1810.07181, Mar. 2021, [Online] https://arxiv.org/abs/1810.07181
%
% Copyright (c) 2021: Zhongyuan Zhao
% Houston, Texas, United States
% <zhongyuan.zhao@huskers.unl.edu>
%% Start of the code

dx = 0.1;
dy = 0.1;
for nbit = 1:4    
    M = 2.^nbit;
    Data = [0:M-1];
    const = qammod(Data,M,'gray');
    bits = de2bi(Data);
    b = num2str(bits); 
    txtBIT = cellstr(b);
    const_re = real(const);
    const_im = imag(const);
    b = num2str(const.'); 
    txtIQ = cellstr(b);
    scatter(const_re, const_im);
    text(const_re+dx,const_im+dy,txtBIT);
    text(const_re+dx,const_im-dy,txtIQ);
    amp = abs(const);
    maxamp = max(amp(:));
    xlim([-maxamp,maxamp]);
    ylim([-maxamp,maxamp]);
end
