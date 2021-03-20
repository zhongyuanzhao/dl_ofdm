%% Compute RMS delay spread of a given power delay profile
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
function [Trms, Tmean] = rms_delay_spread(tau, pdb)
tau = tau(:).';
pdb = pdb(:).';
pli = 10.^( pdb./10);
Tmean = (tau*pli.')./sum(pli);
terrs = (tau-Tmean).^2;
Trms = sqrt((terrs*pli.')./sum(pli));

end