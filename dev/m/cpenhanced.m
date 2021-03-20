%% CP-enhanced channel estimation
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
% The original algorithm from 
% A. A. Quadeer, “Enhanced equalization in OFDM systems using cyclic prefix,” in 
% 2010 IEEE International Conference on Wireless Communications, Networking and 
% Information Security, June 2010, pp. 40–44
%% Start of the code

function [Xdata, Glsfr]=cpenhanced(Rx_Freq, Gls, y, Frame_size, N, L)
    Qinv = conj(dftmtx(N))/N;
%     QNL1 = Qinv(N-L:N, :);
    QNL1 = Qinv(N-L+1:N, :);
    Rx_Freq = reshape(Rx_Freq,[], Frame_size);
    % Gls = repmat(Gls, 1, Frame_size);
    y = reshape(y,[], Frame_size);
    Xls_Freq = Rx_Freq./Gls;
    xls_time = ifft(Xls_Freq);
    ycp_time = y(1:L,:);
    zeropad = zeros(size(ycp_time,1),1);
    xcp_prev = zeros(size(ycp_time,1),1);
    Xdata = zeros(size(Rx_Freq));
    for j = 1:Frame_size
        xcp_this = xls_time(N-L+1:N, j);
        Xcp_ = circshift_comb(xcp_prev, xcp_this, L);
%         Xcp_l= [tril(Xcp_(:,1:L)), zeropad];
%         Xcp_u= [zeropad, triu(Xcp_(:,2:L+1),1)];
        Xcp_l= tril(Xcp_(:,1:L));
        Xcp_u= [zeropad, triu(Xcp_(:,2:L),1)];
%         Xcp_u= Xcp_u(:,1:L);
        ycp = ycp_time(:, j);
%         h_ = pinv(Xcp_'*Xcp_)*Xcp_'*ycp;
        if rcond(Xcp_'*Xcp_) < 1e-10
            h_ = pinv(Xcp_'*Xcp_)*Xcp_'*ycp;
        else
            h_ = inv(Xcp_'*Xcp_)*Xcp_'*ycp;
        end
        H_L = circshift_comb(zeros(size(h_)), h_, L);
        B = [diag(Gls(:,j)); H_L*QNL1];
        C = [Rx_Freq(:,j); ycp - Xcp_u * h_];
        if rcond(B'*B) < 1e-10
            Xesti = pinv(B'*B)*B'*C;
        else
            Xesti = inv(B'*B)*B'*C;
        end
        xcp_prev = xcp_this;
        Xdata(:, j)=Xesti;
    end
    Glsfr = Rx_Freq./Xdata;
end


function mtx = circshift_comb(vec_u, vec_l, L)
    xcp_l = repmat(vec_l(1:L), 1, L+1);
    xcp_u = repmat(vec_u(1:L), 1, L+1);
    Xcp_ul = [xcp_u; xcp_l];
    for ns = 2:L+1
        Xcp_ul(:, ns)=circshift(Xcp_ul(:,ns),ns-1);
    end
    mtx = Xcp_ul(1+L:2*L, :);
    mtx = mtx(:,1:L);
end

