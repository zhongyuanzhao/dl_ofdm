%% Benchmarks for OFDM communication systems in AWGN and Rayleigh channel
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
%% Usage: configurations are hard-coded 
% see script_rayleigh.m for calling this function
% see inline configurations
function OFDM_Benchmark_dev(pilot_type, channel, longcp, eq_idx)
save_tx_data = 0;
N = 64;     % FFT size, Number of total subcarriers
%% Step 1: Configurations
% mobile = ''; % uncomment this to disable mobility
mobile = '_mobile'; % uncomment this to enable mobility
mod_names = {'BPSK','QPSK','8QAM','16QAM'};
est_names = {'Perfect','LS-Spline','LS-Linear','LMMSE','LS-Quadeer',...
    'LMMSE-Quadeer', 'ALMMSE','LMMSE-Uni-PDP', 'LMMSE-Exp-PDP', 'LMMSE-Fast'};
if longcp
    mat_name = sprintf('BER_OFDM_%s_%s_%s_%d_Gray%s.mat',channel,est_names{eq_idx}, pilot_type, N, mobile);
    csv_name = sprintf('BER_OFDM_%s_%s_%s_%d_Gray%s.csv',channel,est_names{eq_idx}, pilot_type, N, mobile);
    fig_name = sprintf('BER_OFDM_%s_%s_%s_%d_Gray%s.fig',channel,est_names{eq_idx}, pilot_type, N, mobile);
    png_name = sprintf('BER_OFDM_%s_%s_%s_%d_Gray%s.png',channel,est_names{eq_idx}, pilot_type, N, mobile);
else
    mat_name = sprintf('BER_OFDM_%s_%s_%s_%d_Gray%s_shortcp.mat',channel,est_names{eq_idx}, pilot_type, N, mobile);
    csv_name = sprintf('BER_OFDM_%s_%s_%s_%d_Gray%s_shortcp.csv',channel,est_names{eq_idx}, pilot_type, N, mobile);
    fig_name = sprintf('BER_OFDM_%s_%s_%s_%d_Gray%s_shortcp.fig',channel,est_names{eq_idx}, pilot_type, N, mobile);
    png_name = sprintf('BER_OFDM_%s_%s_%s_%d_Gray%s_shortcp.png',channel,est_names{eq_idx}, pilot_type, N, mobile);
end
if isfile(mat_name)
    fprintf('%s already exist, skip \n', mat_name)
    return;
else
    fprintf('%s running \n', mat_name)
end

p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    poolsize = 0;
else
    poolsize = p.NumWorkers;
end
% global n_cores;
% poolsize=n_cores;

% if save_tx_data
%     mkdir('mat');
% end
% 1.1 OFDM parameters
if longcp
    Ncp = round(N.*0.25);                               % Length of Cyclic prefix
    cpstr = '';
else
    Ncp = round(N.*0.07);
    cpstr = '_shortcp';
end
if N == 64
    Fs = 960000;
    Np = 6;                                                 % No of pilot symbols
    n_RB = 4;
elseif N==128
    Fs = 1920000;
    Np = 12;                                                 % No of pilot symbols
    n_RB = 6;
elseif N==256
    Fs = 3840000;
    Np = 30;                                                 % No of pilot symbols
    n_RB = 15;
elseif N==512
    Fs = 7680000;
    Np = 50;                                                 % No of pilot symbols
    n_RB = 25;
elseif N==1024
    Fs = 15360000;
    Np = 100;                                                 % No of pilot symbols
    n_RB = 50;
elseif N==1536
    Fs = 23040000;
    Np = 150;                                                 % No of pilot symbols
    n_RB = 75;
elseif N==2048
    Fs = 30720000;
    Np = 300;                                                 % No of pilot symbols
    n_RB = 150;
end
Np=n_RB*2;
Ts = 1/Fs;                                      % Sampling period of channel
Fd = 70;
if strcmp(channel, 'AWGN') || strcmp(channel, 'Flat')
    Fd = 0;                                                 % Max Doppler frequency shift
elseif strcmp(channel, 'EPA') 
    Fd = 5;
elseif strcmp(channel, 'EVA')
    Fd = 70;
elseif strcmp(channel, 'ETU')
    Fd = 70;
elseif strcmpi(channel, 'Custom')
    Fd = 80;
end
if strcmpi(mobile, '') 
    Fd = 0;
end
Ndc = 2;                                                % No of DC guard subcarriers
Ng = (N-Ndc-n_RB.*12)/2;                                % No of side guard subcarriers
% Ndata = N - Np - 2.*Ng - Ndc;                           % No of Data subcarriers per symbol
Frame_size = 7;                                         % OFDM symbols per frame
Nframes = 20000;                                         % Size of tested OFDM frames: set as 10^4 for smooth curve
M_set = [2, 4, 8, 16];                                  % Modulation orders
SNRs = -10:5:30;                                        % Test SNR points
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
pilot_gt = zeros(Np,Frame_size);
pilot_gf = zeros(Np,Frame_size);
for i_sym = 0:Frame_size-1
    haspilot = 0;
    if strcmpi(pilot_type, 'lte') % scattered LTE
        if i_sym==0 
            pilot_sc_sym = Effec_sc([1:6:length(Effec_sc)]); 
            haspilot = 1;
        elseif i_sym==4
            pilot_sc_sym = Effec_sc([4:6:length(Effec_sc)]); 
            haspilot = 1;
        end
    elseif strcmpi(pilot_type, 'block')
        if i_sym==0
            pilot_sc_sym = Effect_sc([1:3:length(Effec_sc)]);
            haspilot = 1;
        end
    elseif strcmpi(pilot_type, 'comb')
        pilot_sc_sym = Effect_sc([1:6:length(Effec_sc)]);
        haspilot = 1;
    else % scattered legacy
        pilot_sc_sym = Effec_sc(sort(mod((pilot_loc + i_sym*3)-1,length(Effec_sc))+1)); 
        haspilot = 1;
    end
    if haspilot
        pilot_gt(:,i_sym+1) = i_sym + 1;
        pilot_gf(:,i_sym+1) = pilot_sc_sym;
        pilot_sc_frame = [pilot_sc_frame, pilot_sc_sym+i_sym*N];
    end
    guard_sc_frame = [guard_sc_frame, guard_sc+i_sym*N];
    DC_sc_frame = [DC_sc_frame, DC_sc+i_sym*N];
end
data_sc_frame = setdiff([1:Frame_size*N],guard_sc_frame);
data_sc_frame = setdiff(data_sc_frame, pilot_sc_frame);
data_sc_frame = setdiff(data_sc_frame, DC_sc_frame);
Ndata_frame = length(data_sc_frame);
pilot_gt = nonzeros(reshape(pilot_gt,Np*Frame_size,1));
pilot_gf = nonzeros(reshape(pilot_gf,Np*Frame_size,1));
[gt,gf] = meshgrid(1:Frame_size,1:N);

% 1.4 Channel
if ~strcmpi(channel, 'AWGN')
    if strcmpi(channel, 'EPA')
        tau = [0, 30, 70, 90, 110, 190, 410].*1e-9;                % Path delays
        pdb = [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8];   % Avg path power gains
        % eq_idx = 2;                                          % LS-Spline
    elseif strcmpi(channel, 'EVA')
        tau = [0, 30, 150, 310, 370, 710, 1090, 1730, 2510].*1e-9; % Path delays
        pdb = [0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9]; % Avg path power gains
        % eq_idx = 2;                                          % LS-Spline
    elseif strcmpi(channel, 'ETU')
        tau = [0, 50, 120, 200, 230, 500, 1600, 2300, 5000].*1e-9; % Path delays
        pdb = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0]; % Avg path power gains
        % eq_idx = 2;                                          % LS-Spline
    elseif strcmpi(channel, 'Custom')
        tau = [0, 70, 200, 230, 500, 1600, 2700, 3000].*1e-9; % Path delays
        pdb = [0.0, -1.4, -1.4, -1.0, -3.0, -9.1, -15.0, -19.0]; % Avg path power gains
    elseif strcmpi(channel, 'Flat')
        tau = [0.0];
        pdb = [0.0];
        % eq_idx = 3;                                          % LS-Linear
    else
        error('Unsupported Channel Estimator Option');
    end
    [Trms, Tmean] = rms_delay_spread(tau, pdb);
    Trms = Trms./Ts;
    Rhh_uni = mmse_pdp(length(tau), N, Trms, 1);
    Rhh_exp = mmse_pdp(length(tau), N, Trms, 0);
%     hdc = parallel.pool.Constant(h);
%     chan = comm.RayleighChannel(...
%         'SampleRate',Fs, ...
%         'PathDelays',tau, ...
%         'AveragePathGains',pdb, ...
%         'NormalizePathGains',true, ...
%         'MaximumDopplerShift',0, ...
%         'RandomStream','Global stream', ...
%         'PathGainsOutputPort',true);
end
Lmst=32;
lsnr20 = 10.^(20/10);
%% Step 2: Test Loops
snr = SNRs;
betas = [1,1,17/9,17/9];
% EsNo= EbNo + 10*log10((N-2.*Np)/N)+ 10*log10(N/(N+Ncp));      % symbol to noise ratio
% snr= EsNo - 10*log10(N/(N+Ncp)); 
if ~strcmp(channel,'AWGN')
    modulations = [1];
else
    modulations = [1,2,3,4];
end
for m_ary = modulations
    beta = betas(m_ary);
    M = M_set(m_ary);                                                  % No of symbols for PSK modulation
    const = qammod([0:M-1],M,'gray'); % Get constellation
    berofdm = zeros(1,length(snr));
    serofdm = zeros(1,length(snr));    
    % Step 2.1 Transmitter
    tic
    matname_tx = sprintf('TX_all_%s_%s_FFT%d%s.mat',mod_names{m_ary}, channel, N, cpstr);
%     if isfile(matname_tx)
%         load(matname_tx);
%     else
    % 2.1.1 Random bits generation
    % D = round((M-1)*rand(Ndata*Frame_size,Nframes));
    D = round((M-1)*rand(Ndata_frame,Nframes));
    % D_test = reshape(D, Ndata, Frame_size*Nframes);
    D_gray = D; % gray2bin(D_test,'qam',M);
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
    % 2.2.1 Rayleigh Fading
    if ~strcmp(channel,'AWGN')
        G = zeros(N*Frame_size, Nframes);
        % totalFrames = c/Frame_size;
        Ch_Data = zeros(size(Tx_Data));
        slice_idx = [1:ceil(Nframes./poolsize):Nframes];
        slice_size = diff([slice_idx, Nframes+1]);
        Ch_Data_sliced = mat2cell(Ch_Data, [r*Frame_size], slice_size);
        Tx_Data_sliced = mat2cell(Tx_Data, [r*Frame_size], slice_size);
        G_sliced = mat2cell(G, [N*Frame_size], slice_size);
        parfor ip=1:poolsize
            h = rayleighchan(Ts, Fd, tau, pdb);
%             h.MaxDopplerShift = 0;
            h.StoreHistory = 0;
            h.StorePathGains = 1;
            h.ResetBeforeFiltering = 1;                              % reset the filter after each call
            [tmp, sliceFrames] = size(Tx_Data_sliced{ip});
            for j = 1:sliceFrames                               % Channel response updated per frame
                Ch_Data_sliced{ip}(:,j) = filter(h,Tx_Data_sliced{ip}(:,j).');        % Pass through Rayleigh channel
                a = h.PathGains;
                AM = h.channelFilter.alphaMatrix;
%                     release(chan);
%                     rng(slice_idx(ip)+j);
%                     [Rx_symbol, a] = chan(Tx_Data_sliced{ip}(:,j));
%                     Ch_Data_sliced{ip}(:,j) = Rx_symbol;
%                     % a = h.PathGains;
%                     chaninfo = info(chan);
%                     AM = chaninfo.ChannelFilterCoefficients;
%                     g = a(1,:)*AM;
                g = a*AM;                                       % Channel coefficients
                Gfr = fft(g.',N);                               % DFT of channel coefficients
                if h.MaxDopplerShift == 0
                    G_sliced{ip}(:,j) = repmat(Gfr(:), Frame_size, 1);            % Slow Fading, Repeat Channel Coefficients
                else
                    G3d = reshape(Gfr, N, [], Frame_size);
                    G2d = mean(G3d, 2);
                    G_sliced{ip}(:,j) = G2d(:);
                end
            end
        end
        Ch_Data = cell2mat(Ch_Data_sliced);
        G = cell2mat(G_sliced);
    else
        Ch_Data = Tx_Data;
        G = ones(N*Frame_size, Nframes);
    end
    GN = G(1:N,:);
    % Long term channel covariance matrix for fast LMMSE
    GNs = reshape(G, N, [], Frame_size*Nframes);
    Rhhlt = (GNs*GNs')./(Frame_size*Nframes);
%     parsave(matname_tx, D, D_gray, txbits, Dmod, Data, txamp,...
%         pilot_signal,  IFFT_Data,  TxCy, r, c, Tx_Data, Tx_Amp,...
%         Tx_Power, Power_PAPR8, Clip_loc, Clip_Data, Tx_Pow_Freq,...
%         totalFrames, G, Ch_Data);
%     end    
    toc
    for i = 1:length(snr)
        tic
        % create filter object inside parfor to avoid error
        lsnr = 10.^(snr(i)/10);
        % 2.2.2 Add AWGN noise
        y = awgn(Ch_Data,snr(i),'measured');                            
        y1 = reshape(y,r,[]);
        % Step 2.3: OFDM Receiver
        % 2.3.1 Remove cyclic prefix 
        Rx = y1(Ncp+1:r,:);
        % 2.3.2 Transform to Frequency-Domain
        Rx_Freq = (sqrt(N-2*Np)/N)*fft(Rx,N,1);
        Xe_Freq = zeros(N*Frame_size,Nframes);
        % long term LMMSE matrix for fast ALMMSE
        Wfast = Rhhlt*inv(Rhhlt+(beta/lsnr)*eye(N));
        if ~strcmp(channel,'AWGN')
            Rx_Freq = reshape(Rx_Freq,N*Frame_size,[]);
            Hhat = Rx_Freq(pilot_sc_frame,:)./pilot_signal;
            if eq_idx == 1
                Gls = G;
            else
                Gls = zeros(N*Frame_size,Nframes);
                parfor j = 1:Nframes
                    Hhat_fr = Hhat(:,j);
                    if eq_idx == 2
                        % Spline Interpolation for scattered 2D pilot
                        Glsfr = griddata(pilot_gf,pilot_gt,Hhat_fr,gf,gt,'v4');
                    elseif eq_idx == 3
                        % Linear Interpolation for scattered 2D pilot
                        fex = scatteredInterpolant(pilot_gf,pilot_gt,Hhat_fr);
                        Glsfr = fex(gf,gt);
                    elseif eq_idx == 4
                        % ideal LMMSE Estimator
                        H = G(:,j);
                        H = reshape(H, N, Frame_size);
                        HhatLS = griddata(pilot_gf,pilot_gt,Hhat_fr,gf,gt,'v4');
                        Glsfr = zeros(N, Frame_size);
                        for ii = 1: Frame_size
                            Rhh = H(:,ii)*H(:,ii)';
                            W = Rhh*inv(Rhh+(beta/lsnr)*eye(N));
                            Glsfr(:,ii) = W*HhatLS(:,ii);
                        end
                    elseif eq_idx == 7
                        % Approximate LMMSE Estimator
                        HhatLS = griddata(pilot_gf,pilot_gt,Hhat_fr,gf,gt,'v4');
                        HhatLSf = mean(HhatLS,2);
                        Rhh = HhatLSf*HhatLSf';
                        Rhh = Rhh./Frame_size;
                        W = Rhh*inv(Rhh+(beta/lsnr)*eye(N));
                        Glsfr = W*HhatLSf;
                        Glsfr = repmat(Glsfr, 1, Frame_size);
                    elseif eq_idx == 8
                        % LMMSE Estimator Uniform PDP
                        HhatLS = griddata(pilot_gf,pilot_gt,Hhat_fr,gf,gt,'v4');
                        HhatLS = mean(HhatLS,2);
                        Rhh = Rhh_uni;
                        W = Rhh/(Rhh+(beta/lsnr)*eye(N));
                        for ii = 1: Frame_size
                            Glsfr(:,ii) = W*HhatLS(:,ii);
                        end
                    elseif eq_idx == 9
                        % LMMSE Estimator Exponential PDP
                        HhatLS = griddata(pilot_gf,pilot_gt,Hhat_fr,gf,gt,'v4');
                        HhatLS = mean(HhatLS,2);
                        Rhh = Rhh_exp; %HhatLS*HhatLS';
                        W = Rhh/(Rhh+(beta/lsnr)*eye(N));
                        for ii = 1: Frame_size
                            Glsfr(:,ii) = W*HhatLS(:,ii);
                        end
                    elseif eq_idx == 10
                        % fast LMMSE Estimator with long term Rhh
                        HhatLS = griddata(pilot_gf,pilot_gt,Hhat_fr,gf,gt,'v4');
                        Glsfr = zeros(N, Frame_size);
                        for ii = 1: Frame_size
                            Glsfr(:,ii) = Wfast*HhatLS(:,ii);
                        end
                    elseif eq_idx == 5
                        % LS with CP exploitation
                        HhatLS = griddata(pilot_gf,pilot_gt,Hhat_fr,gf,gt,'v4');
                        HhatLSf = mean(HhatLS,2);
                        HhatLSf = repmat(HhatLSf,  1, Frame_size);
                        [Xdata, Glsfr] = cpenhanced(Rx_Freq(:,j), HhatLSf, y(:,j), Frame_size, N, Ncp);
                        Xe_Freq(:, j) = reshape(Xdata, N*Frame_size, 1);
                    elseif eq_idx == 6
                        % Enhance ALMMSE with CP exploitation
                        HhatLS = griddata(pilot_gf,pilot_gt,Hhat_fr,gf,gt,'v4');
                        HhatLS = mean(HhatLS,2);
                        Rhh = HhatLS*HhatLS';
                        W = Rhh*inv(Rhh+(beta/lsnr)*eye(N));
                        Glmmse = zeros(N, Frame_size);
                        for ii = 1: Frame_size
                            Glmmse(:,ii) = W*HhatLS;
                        end                        
                        [Xdata, Glsfr] = cpenhanced(Rx_Freq(:,j), Glmmse, y(:,j), Frame_size, N, Ncp);
                        Xe_Freq(:, j) = reshape(Xdata, N*Frame_size, 1);
                    else
                        error('Unsupported Channel Estimator Option');
                    end
                    Gls(:,j) = reshape(Glsfr,N*Frame_size,1);
                end
            end
%             Gls = reshape(Gls,N,Frame_size*Nframes);
            if eq_idx == 5
                Rx_Freq = Xe_Freq;
            else
                Rx_Freq = Rx_Freq./Gls;            
            end
        end
        % 2.3.3 Reshape to Frame size
        FFT_Data = reshape(Rx_Freq,N*Frame_size,[]);
        % 2.3.4 Extract Data Cells
        FFT_Data = reshape(FFT_Data(data_sc_frame,:), [], Nframes);
        % 2.3.5 Demodulation         
        slice_idx = [1:ceil(Nframes./poolsize):Nframes];
        slice_size = diff([slice_idx, Nframes+1]);
        [r1, c1] = size(FFT_Data);
        FFT_Data_sliced = mat2cell(FFT_Data, [r1], slice_size);
        Rx_Data_sliced = cell(1,poolsize);
        parfor ip=1:poolsize
            Rx_Data_sliced{ip} = qamdemod(FFT_Data_sliced{ip},M,'gray');     
        end
        Rx_Data = cell2mat(Rx_Data_sliced);
        Rx_gray = Rx_Data; % gray2bin(Rx_Data,'qam',M);
        rxbits = de2bi(Rx_gray(:));

        
        % Step 2.4 Collect BER and SER
        [bitErrors,ber] = biterr(txbits,rxbits);
        [symErrors,ser] = symerr(Rx_Data(:),D(:));
        serofdm(i) = ser;
        berofdm(i) = ber;
        % Export data for Test in python
        if save_tx_data
            if longcp
                filename = sprintf('./mat/ofdm_%s_%s_%ddB.mat',lower(channel),mod_names{m_ary},snr(i));
            else
                filename = sprintf('./mat/ofdm_%s_%s_%ddB_shortcp.mat',lower(channel),mod_names{m_ary},snr(i));
            end
            parsave(filename, y, txbits, rxbits);
        end
        toc
    end
    berofdm_all(m_ary+1,:) = berofdm;
    serofdm_all(m_ary+1,:) = serofdm;
end

%% Step 3: Result Presentation: Plot BER
parsave(mat_name, berofdm_all, serofdm_all, mat_name);
csvwrite(csv_name, berofdm_all);

figure;
semilogy(SNRs,berofdm_all(2:5,:),'--x','LineWidth',1);
grid('on');
xlabel('SNR (dB)');
ylabel('BER');
legend(mod_names);

titletxt = sprintf('%s Channel',channel);
title(titletxt);

savefig(fig_name);
saveas(gcf, png_name);
end
