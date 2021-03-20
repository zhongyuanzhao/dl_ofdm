%% Compute channel covariance matrix for uniform and exponential power delay profiles
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
%
% Implementation based on reference:
% K. Hung and D. W. Lin, “Pilot-based LMMSE channel estimation for OFDM systems 
% with powerdelay profile approximation,” IEEE Transactions on Vehicular Technology, 
% vol. 59, no. 1, pp. 150–159, 2010
%% Start of the code
function [Rhh] = mmse_pdp(L, N, Trms, uniform_PDP)
Rhh = zeros(N, N);

if uniform_PDP
    for m = 0 : N - 1
        for n = 0 : N - 1
            if L == 0
                r_mn = exp(- 2 * pi * 1j * L * (m - n) / N);
            else
                r_mn = (1 - exp(- 2 * pi * 1j * L * (m - n) / N)) / (2 * pi * 1j * L * (m - n) / N);
            end
            if m == n
                Rhh(m + 1,n + 1) = 1;
            else
                Rhh(m + 1,n + 1) = r_mn;
            end
        end
    end
else
    for m = 0 : N - 1
        for n = 0 : N - 1
            r_mn = (1 - exp(- L * ((1 / Trms) + (2 * pi * 1j * (m - n))/ N))) / (Trms * (1 - exp(- L / Trms)) * ((1 / Trms) + 2 * pi * 1j * (m - n) / N));
            if m == n
                Rhh(m + 1,n + 1) = 1;
            else
                Rhh(m + 1,n + 1) = r_mn;
            end
        end
    end
end
end