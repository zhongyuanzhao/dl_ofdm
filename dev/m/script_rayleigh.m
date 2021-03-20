%% Script for running the benchmarks
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
folders = {'mat','fig'};
for i = 1:length(folders)
    subfolder = folders{i};
    if ~exist(subfolder, 'dir')
       mkdir(subfolder);
    end
end
%% Initialization
close all;
clear all;
clc

% myCluster = parcluster('local');
% myCluster.NumWorkers = 5;  % 'Modified' property now TRUE
% saveProfile(myCluster);    % 'local' profile now updated,


n_cores=4;
% parpool('local', n_cores);
parfor i=1:4
    fprintf('Process: %d',i)
end

%% Main loop
% select pilot pattern
% pilot_type = 'block';
% pilot_type = 'comb';
% pilot_type = 'scatter';
pilot_type = 'LTE';

est_names = {'Perfect','LS-Spline','LS-Linear','LMMSE','LS-Quadeer',...
    'LMMSE-Quadeer', 'ALMMSE','LMMSE-Uni-PDP', 'LMMSE-Exp-PDP', 'LMMSE-Fast'};

ch_names = {'Flat', 'EVA', 'EPA', 'ETU', 'AWGN', 'Custom'};

longcps = [0, 1];

cfgidx=0;

for cidx = 1:6
    channel = ch_names{cidx};
    for longcp = longcps 
        for eq_idx = [ 1, 2, 4, 5, 7, 10 ] %[1, 2, 4, 5, 7]
            cfgidx = cfgidx + 1;
            cfg(cfgidx).pilot_type = pilot_type;
            cfg(cfgidx).channel = channel;
            cfg(cfgidx).longcp = longcp;
            cfg(cfgidx).eq_idx = eq_idx;
        end
    end
end

for i = 1:cfgidx
    OFDM_Benchmark_dev(cfg(i).pilot_type, cfg(i).channel, cfg(i).longcp, cfg(i).eq_idx);
end

