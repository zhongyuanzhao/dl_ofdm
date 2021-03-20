%% about this code
% demonstration of performance of LS, LMMSE, and fast LMMSE channel estimator
% in OFDM system under an ISI-free setting. This simulation is ISI-free
% because fading process is independently applied to each OFDM symbol in
% the inner loop, no leakage between consecutive OFDM symbols. The ideal
% LMMSE estimator computes the channel covariance matrix for each channel
% realization. The fast-LMMSE avoid matrix inversion by using pre-computed 
% LMMSE matrices Wfast at different SNR points, these matrices are obtained 
% from long-term channel covariance matrix Rhh_lt. 
%
% Modified by: Zhongyuan Zhao
% Original source: https://www.mathworks.com/matlabcentral/fileexchange/41634-channel-estimation-for-ofdm-systems
%
%% orignal header
% simulation for channel estimation techniequs using LS, LMMMSE, and
% computationally efficient LMMMSE methods. 
% Prepared by: Hiren Gami
% Ref: J J Van de Beek, "Synchronization and Channel Estimation in OFDM 
% systems", Ph.D thesis,Sept. 1998 
%%

clc
clear all
nCP = 8;%round(Tcp/Ts);
nFFT = 64; 
NT = nFFT + nCP;
F = dftmtx(nFFT)/sqrt(nFFT);
MC = 1500;
EsNodB = -10:5:30;
snr = 10.^(EsNodB/10);
betas = [1,1,17/9,17/9];
%beta = 17/9;
M = 2;
beta = betas(log2(M));
modObj = modem.qammod(M);
demodObj = modem.qamdemod(M);
L = 5;
ChEstLS = zeros(1,length(EsNodB));
ChEstMMSE = zeros(1,length(EsNodB));
LTChEstMMSE = zeros(1,length(EsNodB));
TD_ChEstMMSE = zeros(1,length(EsNodB));
TDD_ChEstMMSE = zeros(1,length(EsNodB));
TDQabs_ChEstMMSE = zeros(1,length(EsNodB));

% Compute long-term channel covariance matrix by drawing MC realizations
% from the same channel model used in the simulation.
H_lt = [];
for mc = 1:MC
    g = randn(L,1)+1i*randn(L,1);
    g = g/norm(g);
    H_lt(:,mc) = fft(g,nFFT);
end
Rhh_lt = (H_lt*H_lt')./MC;

for ii = 1:length(EsNodB)
    disp('EsN0dB is :'); disp(EsNodB(ii));tic;
    ChMSE_LS = 0;
    ChMSE_LMMSE=0; 
    LTChMSE_LMMSE=0; 
    TDMSE_LMMSE =0;
    TDDMSE_LMMSE=0;
    TDQabsMSE_LMMSE =0;
    % Compute fast LMMSE matrix per SNR point
    Wfast = Rhh_lt/(Rhh_lt+(beta/snr(ii))*eye(nFFT));
    for mc = 1:MC
% Random channel taps
        g = randn(L,1)+1i*randn(L,1);
        g = g/norm(g);
        H = fft(g,nFFT);
% generation of symbol
        X = randi([0 M-1],nFFT,1);  %BPSK symbols
        XD = modulate(modObj,X)/sqrt(10); % normalizing symbol power
        x = F'*XD;
        xout = [x(nFFT-nCP+1:nFFT);x];        
% channel convolution and AWGN
        y = conv(xout,g);
        nt =randn(nFFT+nCP+L-1,1) + 1i*randn(nFFT+nCP+L-1,1);
        No = 10^(-EsNodB(ii)/10);
        y =  y + sqrt(No/2)*nt;
% Receiver processing
        y = y(nCP+1:NT);
        Y = F*y;
        MSEref = mean(var(H,1));
%        NMSE = mse(target-output)/MSEref;        
% frequency doimain LS channel estimation 
        HhatLS = Y./XD; 
        ChMSE_LS = ChMSE_LS + ((H -HhatLS)'*(H-HhatLS))/(nFFT*MSEref);
%         ChMSE_LS = ChMSE_LS + mse(H-HhatLS)/MSEref;
% Frequency domain LMMSE estimation
        Rhh = H*H';
        W = Rhh/(Rhh+(beta/snr(ii))*eye(nFFT));
        HhatLMMSE = W*HhatLS;
        ChMSE_LMMSE = ChMSE_LMMSE + ((H -HhatLMMSE)'*(H-HhatLMMSE))/(nFFT*MSEref);        
%         ChMSE_LMMSE = ChMSE_LMMSE + mse(H -HhatLMMSE)/MSEref;        
% Frequency domain fast LMMSE estimation with long term Rhh
        LT_HhatLMMSE = Wfast*HhatLS;
        LTChMSE_LMMSE = LTChMSE_LMMSE + ((H -LT_HhatLMMSE)'*(H-LT_HhatLMMSE))/(nFFT*MSEref);        
%         LTChMSE_LMMSE = LTChMSE_LMMSE + mse(H -LT_HhatLMMSE)/MSEref;        
% Time domain LMMSE estimation
%         ghatLS = ifft(HhatLS,nFFT);
%         Rgg = g*g';
%         WW = Rgg/(Rgg+(beta/snr(ii))*eye(L));
%         ghat = WW*ghatLS(1:L);
%         TD_HhatLMMSE = fft(ghat,nFFT);%        
%         TDMSE_LMMSE = TDMSE_LMMSE + ((H -TD_HhatLMMSE)'*(H-TD_HhatLMMSE))/nFFT;   
 % Time domain LMMSE estimation - ignoring channel covariance
%         ghatLS = ifft(HhatLS,nFFT);
%         Rgg = diag(g.*conj(g));
%         WW = Rgg/(Rgg+(beta/snr(ii))*eye(L));
%         ghat = WW*ghatLS(1:L);
%         TDD_HhatLMMSE = fft(ghat,nFFT);%        
%         TDDMSE_LMMSE = TDDMSE_LMMSE + ((H -TDD_HhatLMMSE)'*(H-TDD_HhatLMMSE))/nFFT;    
  
  % Time domain LMMSE estimation - ignoring smoothing matrix
%         ghatLS = ifft(HhatLS,nFFT);
%         TDQabs_HhatLMMSE = fft(ghat,nFFT);%        
%         TDQabsMSE_LMMSE = TDQabsMSE_LMMSE + ((H -TDQabs_HhatLMMSE)'*(H-TDQabs_HhatLMMSE))/nFFT;          
         
    end
    ChEstLS(ii) = ChMSE_LS/MC;
    ChEstMMSE(ii)=ChMSE_LMMSE/MC;
    LTChEstMMSE(ii)=LTChMSE_LMMSE/MC;
%     TD_ChEstMMSE(ii)=TDMSE_LMMSE/MC;
%     TDD_ChEstMMSE(ii)=TDDMSE_LMMSE/MC;
%     TDQabs_ChEstMMSE(ii)=TDQabsMSE_LMMSE/MC;
    toc;
end
% Channel estimation 
ChEstLS_dB = 10.*log10(ChEstLS);
ChEstMMSE_dB = 10.*log10(ChEstMMSE);
LTChEstMMSE_dB = 10.*log10(LTChEstMMSE);
plot(EsNodB,ChEstLS_dB,'r','LineWidth',1,'Marker','v');
hold on;grid on;xlabel('EbNo(dB)'); ylabel('NMSE (dB)');
plot(EsNodB,ChEstMMSE_dB,'k','LineWidth',1,'Marker','d');
plot(EsNodB,LTChEstMMSE_dB,'g','LineWidth',1,'Marker','s');
% semilogy(EsNodB,TDD_ChEstMMSE,'m','LineWidth',2);
% semilogy(EsNodB,TDQabs_ChEstMMSE,'b','LineWidth',2);
% Theoratical bound calculation
% semilogy(EsNodB,beta./snr,'-.r*','LineWidth',2);
% ThLMMSE = (1/nFFT)*(beta./snr).*(1./(1+(beta./snr)));
% semilogy(EsNodB,ThLMMSE,'-.k*','LineWidth',2);
hold off;
% legend('LS','MMSE', 'Long term LMMSE','Theory-LS', 'Theory-LMMSE');
legend('LS','ideal-LMMSE', 'Fast-ALMMSE');
xticks([-10:5:30]);
set(gca,'Position',[0.12,0.12,0.86,0.85]);
set(gca,'FontSize',12);
set(gcf,'PaperUnits','points','PaperPosition',[26,186,426,335]);
