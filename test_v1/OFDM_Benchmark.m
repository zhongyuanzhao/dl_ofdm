% Benchmark code for OFDM communication system in AWGN channel
% Author: Zhongyuan Zhao, 
% Email: zhongyuan.zhao@huskers.unl.edu
% Date: 2018-10-10

%% Step 1: Configurations
clc;
%clear all;
close all;
save_tx_data = 1;
if save_tx_data
    mkdir('mat');
end
% 1.1 OFDM parameters
N = 64;                                                 % FFT size, Number of total subcarriers
Ncp = 16;                                               % Length of Cyclic prefix
Ts = 1e-7;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 8;                                                 % No of pilot symbols
Ng = 4;                                                 % No of side guard subcarriers
Ndc = 2;                                                % No of DC guard subcarriers
Ndata = N - Np - 2.*Ng - Ndc;                           % No of Data subcarriers per symbol
Frame_size = 8;                                         % OFDM symbols per frame
Nframes = 2000;                                         % Size of tested OFDM frames: set as 10^4 for smooth curve
M_set = [2, 4, 8, 16];                                  % Modulation orders
SNRs = -10:1:29;                                        % Test SNR points
% 1.2 Vehicles for results
berofdm_all = zeros(5,length(SNRs));
berofdm_all(1,:) = SNRs;
serofdm_all = zeros(5,length(SNRs));
serofdm_all(1,:) = SNRs;
% 1.3 Calculate pilot locations
DC_sc = [N/2, N/2+1];
Effec_sc = [Ng+1:N-Ng];
Effec_sc = setdiff(Effec_sc, DC_sc);
% Pilot_sc = [5,12,19,26,34,41,48,55];
pilot_loc = [1:ceil(length(Effec_sc)/Np):length(Effec_sc)];
Pilot_sc = Effec_sc(pilot_loc);
guard_sc = [1:Ng,N-Ng+1:N];

Np = length(pilot_loc); % Recalculate Number of pilot
pilot_sc_frame = [];
guard_sc_frame = [];
DC_sc_frame = [];
for i_sym = 0:Frame_size-1
    pilot_sc_sym = Effec_sc(sort(mod((pilot_loc + i_sym*3)-1,length(Effec_sc))+1)); % scattered
%     pilot_sc_sym = Pilot_sc; % comb pilot
    pilot_sc_frame = [pilot_sc_frame, pilot_sc_sym+i_sym*N];
    guard_sc_frame = [guard_sc_frame, guard_sc+i_sym*N];
    DC_sc_frame = [DC_sc_frame, DC_sc+i_sym*N];
end
data_sc_frame = setdiff([1:Frame_size*N],guard_sc_frame);
data_sc_frame = setdiff(data_sc_frame, pilot_sc_frame);
data_sc_frame = setdiff(data_sc_frame, DC_sc_frame);

%% Step 2: Test Loops
mod_names = {'BPSK','QPSK','8QAM','16QAM'};
channel = 'AWGN';
mat_name = sprintf('BER_OFDM_%s_Gray.mat',channel);
csv_name = sprintf('BER_OFDM_%s_Gray.csv',channel);
snr = SNRs;
% EsNo= EbNo + 10*log10((N-2.*Np)/N)+ 10*log10(N/(N+Ncp));      % symbol to noise ratio
% snr= EsNo - 10*log10(N/(N+Ncp)); 
for m_ary = 1:4
    M = M_set(m_ary);                                                  % No of symbols for PSK modulation
    const = qammod([0:M-1],M,'gray'); % Get constellation
    berofdm = zeros(1,length(snr));
    serofdm = zeros(1,length(snr));    
    for i = 1:length(snr)
        % Step 2.1 Transmitter
        % 2.1.1 Random bits generation
        D = round((M-1)*rand(Ndata*Frame_size,Nframes));
        D_test = reshape(D, Ndata, Frame_size*Nframes);
        D_gray = D_test; % gray2bin(D_test,'qam',M);
        txbits = de2bi(D_gray(:)); % transmitted bits
        % 2.1.2 Modulation 
        if M == 8
            Dmod = qammod(D,M,'gray');
        else
            Dmod = qammod(D,M,'gray');
        end
        % 2.1.3 Framing
        Data = zeros(N*Frame_size,Nframes);   % Guard sc Insertion
        Data(data_sc_frame,:) = Dmod;   % Data sc Insertion
        txamp = max(abs(Dmod(:)));
        pilot_signal = txamp.*sqrt(1/2).*(1+1i); % Norm pilot power to peak constellation power
        Data(pilot_sc_frame,:)= pilot_signal; % Pilot sc Insertion
        Data = reshape(Data, N, Frame_size*Nframes);
        % 2.1.4 To Time-domain OFDM symbol
        IFFT_Data = (N/sqrt(N-2*Np))*ifft(Data,N);
        TxCy = [IFFT_Data((N-Ncp+1):N,:); IFFT_Data];       % Add Cyclic prefix
        [r, c] = size(TxCy);
        Tx_Data = TxCy;
        % 2.1.5 Clip PAPR to 8 (9dB)
        Tx_Amp = abs(Tx_Data);
        Tx_Power = Tx_Amp.^2;
        Power_PAPR8 = 8.*mean(Tx_Power,1);
        Clip_loc = Tx_Power > Power_PAPR8;
        Clip_Data = Tx_Data./Tx_Amp;
        Clip_Data = sqrt(Power_PAPR8).*Clip_Data;
        Tx_Data(Clip_loc) = Clip_Data(Clip_loc);
        % Step 2.2 Wireless Channel
        Tx_Pow_Freq = mean2(abs(Tx_Data).^2);
        Tx_Data = reshape(Tx_Data, r*Frame_size,[]);
        totalFrames = c/Frame_size;
        % 2.2.2 Add AWGN noise
        y = awgn(Tx_Data,snr(i),'measured');                            
        y = reshape(y,r,[]);
        % Step 2.3: OFDM Receiver
        % 2.3.1 Remove cyclic prefix 
        Rx = y(Ncp+1:r,:);               
        % 2.3.2 Transform to Frequency-Domain
        Rx_Freq = (sqrt(N-2*Np)/N)*fft(Rx,N,1);
        % 2.3.3 Reshape to Frame size
        FFT_Data = reshape(Rx_Freq,N*Frame_size,[]);
        % 2.3.4 Extract Data Cells
        FFT_Data = reshape(FFT_Data(data_sc_frame,:), [], Nframes*Frame_size);
        % 2.3.5 Demodulation 
        if m_ary==3
            Rx_Data = qamdemod(FFT_Data,M,'gray');    
        else
            Rx_Data = qamdemod(FFT_Data,M,'gray');     
        end
        Rx_gray = Rx_Data; % gray2bin(Rx_Data,'qam',M);
        rxbits = de2bi(Rx_gray(:));
        % Step 2.4 Collect BER and SER
        [bitErrors,ber] = biterr(txbits,rxbits);
        [symErrors,ser] = symerr(Rx_Data(:),D(:));
        serofdm(i) = ser;
        berofdm(i) = ber;
        % Export data for Test in python
        if save_tx_data
            filename = sprintf('./mat/ofdm_awgn_%s_%ddB.mat',mod_names{m_ary},snr(i));
            save(filename,'y','txbits','rxbits');
        end
    end
    berofdm_all(m_ary+1,:) = berofdm;
    serofdm_all(m_ary+1,:) = serofdm;
end

%% Step 3: Result Presentation: Plot BER
save(mat_name, 'berofdm_all','serofdm_all');
csvwrite(csv_name, berofdm_all);
figure;
semilogy(SNRs,berofdm_all(2:5,:),'--x','LineWidth',2);
grid on;
title('OFDM BER vs SNR in AWGN channel');
xlabel('SNR (dB)');
ylabel('BER');
legend(mod_names);
