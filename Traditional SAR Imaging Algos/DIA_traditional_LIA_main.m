% Developed by Lhamo Dorje and Dr. Xiaohua (Edward) Li, November, 2025
% ECE department, SUNY Binghamton


%% Differential Imaging Attack on Near-Field SAR Imaging
% This script implements a differential spoofing attack on classical
% near-field SAR imaging algorithms (MFA, RMA, BPA, (advanced) LIA).
%
% Steps:
%   1) Extract an attack waveform pool from paired clean/attacked IQ data.
%   2) Load victim SAR data and target (camouflaged) image.
%   3) Build an attack raw-data model and optimize complex gains A.
%   4) Reconstruct the attacked SAR image and evaluate performance.
%
% Required files on MATLAB path:
%   - iqData_noAtk.mat: iqData [Nsamp x nRX x nFrame] (clean IQ)
%   - iqData_Atk.mat  : iqData [Nsamp x nRX x nFrame] (attacked IQ)
%   - rawSAR.mat      : adcDataCube [Nsamp x M x N] (raw SAR cube)
%   - trueImage_complex_<algo>.mat        : trueImage_complex [Ny x Nx]
%   - desired_attacked_complex_MFA_RMA.mat: sar_camouflaged [Ny x Nx]
%   - desired_attacked_complex_BPA_LIA.mat: sar_camouflaged [Ny x Nx]
%
% where <algo> in {MFA, RMA, BPA, LIA}.

clc; clear; close all;

%% ---------------------------------------------------------------
%  User selection: SAR imaging algorithm
% ---------------------------------------------------------------
sar_algo = 'LIA';            % Options: 'MFA', 'RMA', 'BPA', 'LIA'

%% ---------------------------------------------------------------
%  Step 1: Prepare & Extract Attack Signal Pool (X_aa)
% ---------------------------------------------------------------

% Load IQ data
try
    r0 = load("iqData_noAtk.mat").iqData;   % Nsamp x nRX x nFrame (clean IQ)
    r1 = load("iqData_Atk.mat").iqData;     % Nsamp x nRX x nFrame (attacked IQ)
catch ME
    error('Could not load IQ files. Ensure iqData_noAtk.mat and iqData_Atk.mat are on the path.\nMATLAB Error: %s', ME.message);
end

[Nsamp, nRX, nFrame] = size(r0);
p      = nRX * nFrame;
r0_vec = reshape(r0, Nsamp, p); % Nsamp x p
r1_vec = reshape(r1, Nsamp, p);

% Alignment parameters
tau_max_guess = 20;  % integer delay search window (samples)
tau_search    = max(-tau_max_guess, -(Nsamp-1)) : min(tau_max_guess, Nsamp-1);

r_v_est   = zeros(Nsamp, p, 'like', r0_vec);     % estimated victim (alpha * shifted r0)
r_a_est   = zeros(Nsamp, p, 'like', r0_vec);     % extracted attack residual
alpha_vec = complex(zeros(1, p, 'like', r0_vec)); % complex LS scaling factor
tau_samps = zeros(1, p);                         % selected integer delay

fprintf('Running exhaustive delay search + LS scaling on p=%d channels...\n', p);
tic;

% Main alignment loop
for col = 1:p
    r0_col = r0_vec(:, col);
    r1_col = r1_vec(:, col);

    % Skip if clean signal is near zero
    if norm(r0_col) < eps
        r_a_est(:, col) = r1_col;
        continue;
    end

    best_err         = Inf;
    best_alpha       = 0;
    best_tau         = 0;
    best_r0_shifted  = zeros(Nsamp, 1, 'like', r0_col);

    % Exhaustive integer delay search + LS scaling
    for tau = tau_search
        % zero-padded, non-circular shift
        if tau > 0
            r0_shifted = [zeros(tau, 1, 'like', r0_col); r0_col(1:end-tau)];
        elseif tau < 0
            t = -tau;
            r0_shifted = [r0_col(t+1:end); zeros(t, 1, 'like', r0_col)];
        else
            r0_shifted = r0_col;
        end

        denom = (r0_shifted' * r0_shifted);
        alpha = (r0_shifted' * r1_col) / (denom + eps);

        err = sum(abs(r1_col - alpha * r0_shifted).^2);

        if err < best_err
            best_err        = err;
            best_alpha      = alpha;
            best_tau        = tau;
            best_r0_shifted = r0_shifted;
        end
    end

    % Store best alignment result
    alpha_vec(col)   = best_alpha;
    tau_samps(col)   = best_tau;
    r_v_est(:, col)  = best_alpha * best_r0_shifted;
    r_a_est(:, col)  = r1_col - r_v_est(:, col);
end

toc;
fprintf('Finished alignment. mean|alpha| = %.3e\n', mean(abs(alpha_vec)));

% Build attack pool X_aa (Nsamp x p^2), using all pairs (i, j)
vhat = r_v_est;           % vhat(:, i) = alpha_i * shifted r0(:, i)
p2   = p * p;
X_aa = zeros(Nsamp, p2, 'like', r0_vec);
k    = 1;

for i = 1:p
    v_i = vhat(:, i);
    for j = 1:p
        X_aa(:, k) = r1_vec(:, j) - v_i;
        k = k + 1;
    end
end

% Column-wise RMS normalization for X_aa
colrms = sqrt(mean(abs(X_aa).^2, 1));
X_aa   = X_aa ./ (colrms + eps);

stat = @(X) [min(abs(X(:))), max(abs(X(:))), mean(abs(X(:)))];
fprintf('X_aa: size = %d x %d | min=%.3e max=%.3e mean=%.3e\n', ...
    size(X_aa, 1), size(X_aa, 2), stat(X_aa));

% Optional: save attack pool
% save('X_aa.mat','X_aa');

%% ---------------------------------------------------------------
%  Step 2: Load Victim SAR Data and Setup Imaging Parameters
% ---------------------------------------------------------------

% Filenames for clean and desired attacked images
filename_clean = sprintf('trueImage_complex_%s.mat', sar_algo);

switch upper(sar_algo)
    case {'MFA', 'RMA'}
        filename_attacked = 'desired_attacked_complex_MFA_RMA.mat';
    case {'BPA', 'LIA'}
        filename_attacked = 'desired_attacked_complex_BPA_LIA.mat';
    otherwise
        error('Unknown SAR algorithm: %s', sar_algo);
end

try
    % Raw SAR cube (victim measurements)
    sarRawData = load('rawSAR.mat').adcDataCube;      % Nsamp x M x N

    % Clean complex SAR image (reference)
    S_clean           = load(filename_clean);
    trueImage_complex = S_clean.trueImage_complex;

    % Desired attacked complex image (target)
    S_attacked               = load(filename_attacked);
    desired_attacked_complex = S_attacked.sar_camouflaged;

    % Handle possible dlarray inputs
    if isa(trueImage_complex, 'dlarray')
        trueImage_complex = extractdata(trueImage_complex);
    end
    if isa(desired_attacked_complex, 'dlarray')
        desired_attacked_complex = extractdata(desired_attacked_complex);
    end

catch ME
    error(['Could not load SAR data for algorithm %s. ', ...
           'Check that rawSAR.mat, %s, and %s are on the path.\nMATLAB Error: %s'], ...
           sar_algo, filename_clean, filename_attacked, ME.message);
end

disp(['Successfully loaded data for SAR algorithm: ', sar_algo]);

[Nsamp_sar, M, N] = size(sarRawData);
assert(Nsamp_sar == Nsamp, 'Sample count mismatch between IQ and SAR data.');

X_v = reshape(sarRawData, Nsamp, M * N);   % Nsamp x Np
Np  = M * N;

% SAR system parameters
c0 = physconst('lightspeed');
F0 = 77e9;          % start frequency (Hz)
FS = 5000e3;        % sampling rate (samples/s)
Ts = 1 / FS;        % sampling period
K0 = 70.295e12;     % chirp slope (Hz/s)
tI = 4.5225e-10;    % instrument delay (s)

nFFTtime  = 1024;   % range-FFT points
nFFTspace = 1024;   % spatial FFT points (MFA/RMA)

%% ---------------------------------------------------------------
%  Step 2.1: Algorithm-specific imaging setup (Data Pre-processing)
% ---------------------------------------------------------------

switch upper(sar_algo)
    case 'MFA'
        dx   = 1;                         % horizontal step (mm)
        dy   = 1;                         % vertical step (mm)
        bbox = [-200 200 -200 200];       % [xmin xmax ymin ymax] in mm
        z0   = 185;                       % target range (mm)

        k0_range_bin = round(K0 / FS * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        rawDataFFT = fft(sarRawData, nFFTtime);
        sarData    = squeeze(rawDataFFT(k0_range_bin + 1, :, :)); % M x N

        % Serpentine correction
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        [~, ~, trueImage_abs, ~] = dlMFA(sarData, params);
        trueImage_abs = extractdata(trueImage_abs);

        plot_sar(trueImage_abs, bbox, dx, dy, 'Clean Reconstructed Image (MFA)');

    case 'RMA'
        Echo = permute(sarRawData, [3, 2, 1]);   % [samples, vertical, horizontal] -> [horizontal, vertical, samples]
        dx   = 1;
        dy   = 1;
        bbox = [-200 200 -200 200];

        Nx          = 200;
        Nz          = 200;
        num_sample  = size(Echo, 3);
        nFFTtime    = num_sample;
        rawDataFFT  = fft(Echo, nFFTtime, 3);

        ID_select   = 6; % example index (application-specific)
        k0_range_bin = ID_select;

        sarData = squeeze(rawDataFFT(:, :, ID_select)).';
        z0      = c0 / 2 * (ID_select / (K0 * (1/FS) * nFFTtime) - tI);

        for ii = 2:2:Nz
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        [~, ~, trueImage_abs, ~] = dlRMA(sarData, params);
        trueImage_abs = extractdata(trueImage_abs);

        plot_sar(trueImage_abs, bbox, dx, dy, 'Clean Reconstructed Image (RMA)');

    case 'BPA'
        dx   = 1;
        dy   = 1;
        bbox = [-200 200 -200 200];
        z0   = 185;

        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * Ts * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);
        sarData      = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        A = 50;   % horizontal pixels
        B = 50;   % vertical pixels

        params = struct('z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'Nsamp', Nsamp, 'nFFTtime', nFFTtime, 'N', N, 'M', M, ...
                        'A_bpa', A, 'B_bpa', B, 'F0', F0, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        H_bpa        = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        [~, ~, trueImage_abs, ~] = dlBPA(sarData, params, H_bpa);
        trueImage_abs = extractdata(trueImage_abs);

    case 'LIA'
        dx   = 1;
        dy   = 1;
        bbox = [-200 200 -200 200];
        z0   = 185;

        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * Ts * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);
        sarData      = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        A = 50;
        B = 50;

        params = struct('z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'Nsamp', Nsamp, 'nFFTtime', nFFTtime, 'N', N, 'M', M, ...
                        'A_bpa', A, 'B_bpa', B, 'F0', F0, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        H_bpa        = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        NM       = M * N;
        kk       = min(40000, NM);
        rng(1000);
        params.py = sort(randperm(NM, kk));

        [~, ~, trueImage_abs, ~] = dlLIA(sarData, params, H_bpa);
        trueImage_abs = extractdata(trueImage_abs);

    otherwise
        error('Invalid SAR algorithm selection.');
end

%% ---------------------------------------------------------------
%  Normalize target images (for MSE loss consistency)
% ---------------------------------------------------------------

true_abs = abs(trueImage_complex);
atk_abs  = abs(desired_attacked_complex);

rms_true = sqrt(mean(true_abs(:).^2) + eps);
rms_atk  = sqrt(mean(atk_abs(:).^2)  + eps);

params.trueImage       = dlarray(true_abs ./ rms_true, "SS");
params.desiredAtkImage = dlarray(atk_abs  ./ rms_atk,  "SS");

algo_name = upper(params.sar_algo);

% Sanity-check visualization
figure('Name', ['Target & Reference Images (Algo: ', algo_name, ')'], ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.5]);

dx   = params.dx;
dy   = params.dy;
bbox = params.bbox;

sarImage_clean   = extractdata(params.trueImage);
sarImage_true    = extractdata(params.trueImage);
sarImage_desired = extractdata(params.desiredAtkImage);

top_val = max([sarImage_true(:); sarImage_desired(:)]);
clim    = [0, top_val];

if ismember(algo_name, {'BPA','LIA'})
    subplot(1,3,1);
    plot_sar_bpa(sarImage_clean, bbox, dx, dy, ...
        sprintf('1. Clean Reconstructed Image (%s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

    subplot(1,3,2);
    plot_sar_bpa(sarImage_true, bbox, dx, dy, ...
        sprintf('2. Loaded True Image (%s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

    subplot(1,3,3);
    plot_sar_bpa(sarImage_desired, bbox, dx, dy, ...
        sprintf('3. Desired Attacked Image (%s)', algo_name));
    caxis(clim); colorbar; colormap(jet);
else
    subplot(1,3,1);
    plot_sar(sarImage_clean, bbox, dx, dy, ...
        sprintf('1. Clean Reconstructed Image (%s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

    subplot(1,3,2);
    plot_sar(sarImage_true, bbox, dx, dy, ...
        sprintf('2. Loaded True Image (%s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

    subplot(1,3,3);
    plot_sar(sarImage_desired, bbox, dx, dy, ...
        sprintf('3. Desired Attacked Image (%s)', algo_name));
    caxis(clim); colorbar; colormap(jet);
end

%% ---------------------------------------------------------------
%  Step 3: Build Attack Waveforms and Optimize Complex Gains A
% ---------------------------------------------------------------

% 3.1 Select and frequency-align attack waveforms
targetK   = 40000;                     % attack pool size to sample
rng(0);
sample_idx = randi(p * p, [1, targetK]);
X_a_pool   = X_aa(:, sample_idx);

sel_idx = randi(size(X_a_pool, 2), [1, Np]); % one waveform per aperture
X_a     = X_a_pool(:, sel_idx);              % Nsamp x Np

% Per-column frequency shift to victim range bin
Xspec   = fft(X_a, nFFTtime, 1);               % Nsamp x Np
[~, b0] = max(abs(Xspec), [], 1);
f0      = (b0 - 1) * FS / nFFTtime;
f_tgt   = (params.k0_range_bin) * FS / nFFTtime;
Delta   = f0 - f_tgt;

t = (0:Nsamp-1).' / FS;
P = exp(-1j * 2*pi * (t * Delta));             % Nsamp x Np
D = P .* X_a;                                  % Nsamp x Np

% Scale D to match victim RMS per column
colrms = @(X) sqrt(mean(abs(X).^2, 1));
scale  = (colrms(X_v) + eps) ./ (colrms(D) + eps);
D      = D .* scale;

% 3.2 Optimization setup (algorithm-specific learning rates)
switch upper(sar_algo)
    case 'MFA'
        maxIter   = 300; 
        lr_re     = 1e2;  
        lr_im     = 1e3;  
        lambda_L2 = 1e-4;
    case 'RMA'
        maxIter   = 300; 
        lr_re     = 1e3;  
        lr_im     = 1e2;  
        lambda_L2 = 1e-4;
    case 'BPA'
        maxIter   = 10;  
        lr_re     = 1e3;  
        lr_im     = 1e3;  
        lambda_L2 = 1e-4;
    case 'LIA'
        maxIter   = 300; 
        lr_re     = 1e3;  
        lr_im     = 1e3;  
        lambda_L2 = 1e-4;
    otherwise
        error('Invalid SAR algorithm selection.');
end

use_projection = true;
Amax           = 2;

A_re = dlarray(1e-3 * randn(Np, 1, 'double'));
A_im = dlarray(1e-3 * randn(Np, 1, 'double'));

loss_hist  = zeros(maxIter, 1);
meanA_hist = zeros(maxIter, 1);
maxA_hist  = zeros(maxIter, 1);

% Live plots
figure('Name','Attack Optimization Progress','Color','w');

subplot(3,1,1);
hLoss = semilogy(nan, nan, '-o');
grid on; xlabel('Iteration'); ylabel('Loss'); title('Loss vs Iteration');

subplot(3,1,2);
hMeanA = plot(nan, nan, '-o');
grid on; xlabel('Iteration'); ylabel('mean|A|'); title('Mean |A| vs Iteration');

subplot(3,1,3);
hMaxA = plot(nan, nan, '-o');
grid on; xlabel('Iteration'); ylabel('max|A|'); title('Max |A| vs Iteration');

fprintf('\nOptimizing complex gain A for all locations (Np=%d) with SAR algorithm: %s\n', ...
    Np, params.sar_algo);

for iter = 1:maxIter
    [loss, gRe, gIm] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);

    lossVal          = double(gather(extractdata(loss)));
    loss_hist(iter)  = lossVal;

    % Gradient steps
    A_re = A_re - lr_re * gRe;
    A_im = A_im - lr_im * gIm;

    % Magnitude projection on |A| <= Amax
    if use_projection
        A_num = extractdata(A_re) + 1j * extractdata(A_im);
        mags  = abs(A_num);
        over  = mags > Amax;
        if any(over)
            scale_proj      = ones(size(mags));
            scale_proj(over) = Amax ./ mags(over);
            A_num           = A_num .* scale_proj;
            A_re            = dlarray(real(A_num));
            A_im            = dlarray(imag(A_num));
        end
    end

    A_now            = extractdata(A_re) + 1j * extractdata(A_im);
    meanA_hist(iter) = mean(abs(A_now));
    maxA_hist(iter)  = max(abs(A_now));

    iters = 1:iter;

    set(hLoss,  'XData', iters, 'YData', loss_hist(1:iter));
    set(hMeanA, 'XData', iters, 'YData', meanA_hist(1:iter));
    set(hMaxA,  'XData', iters, 'YData', maxA_hist(1:iter));
    drawnow limitrate;

    if mod(iter, 2) == 0 || iter == 1 || iter == maxIter
        fprintf('Iter %3d/%3d | Loss=%.6e | mean|A|=%.3e, max|A|=%.3e\n', ...
            iter, maxIter, lossVal, meanA_hist(iter), maxA_hist(iter));
    end
end
fprintf('-----------------------------\n');

%% ---------------------------------------------------------------
%  Step 3.3: Reconstruct Attacked Image with Optimal A
% ---------------------------------------------------------------

A_opt   = extractdata(A_re) + 1j * extractdata(A_im);
Y_opt   = X_v + D .* (A_opt.');          % Nsamp x Np
Y_cube  = reshape(Y_opt, Nsamp, M, N);
rawDataFFT_att = fft(Y_cube, nFFTtime);
sarData_att    = squeeze(rawDataFFT_att(k0_range_bin + 1, :, :));

for ii = 2:2:size(sarData_att, 1)
    sarData_att(ii, :) = fliplr(sarData_att(ii, :));
end

switch upper(params.sar_algo)
    case 'MFA'
        [~, ~, atkImage_abs, ~] = dlMFA(sarData_att, params);
        I_att = gather(extractdata(atkImage_abs));
        figure('Name','Attacked Image (MFA)','Color','w');
        plot_sar(I_att, bbox, dx, dy, 'Attacked Image (MFA)');
        caxis(clim); colorbar; colormap(jet);

    case 'RMA'
        [~, ~, atkImage_abs, ~] = dlRMA(sarData_att, params);
        I_att = gather(extractdata(atkImage_abs));
        figure('Name','Attacked Image (RMA)','Color','w');
        plot_sar(I_att, bbox, dx, dy, 'Attacked Image (RMA)');
        caxis(clim); colorbar; colormap(jet);

    case 'BPA'
        [~, ~, atkImage_abs, ~] = dlBPA(sarData_att, params, params.H_bpa);
        I_att = gather(extractdata(atkImage_abs));
        figure('Name','Attacked Image (BPA)','Color','w');
        plot_sar_bpa(I_att, params.bbox, params.dx, params.dy, ...
            'Attacked Image (BPA)');
        caxis(clim); colorbar; colormap(jet);

    case 'LIA'
        [~, ~, atkImage_abs, ~] = dlLIA(sarData_att, params, params.H_bpa);
        I_att = gather(extractdata(atkImage_abs));
        figure('Name','Attacked Image (LIA)','Color','w');
        plot_sar_bpa(I_att, params.bbox, params.dx, params.dy, ...
            'Attacked Image (LIA)');
        caxis(clim); colorbar; colormap(jet);

    otherwise
        error('Invalid SAR algorithm selection for final reconstruction.');
end

% Optional: export attacked image for LaTeX / slides
% exportgraphics(gcf, sprintf('sar_%s_attacked_figure.pdf', params.sar_algo), 'ContentType','vector');
% exportgraphics(gcf, sprintf('sar_%s_attacked_figure.png', params.sar_algo), 'Resolution', 600);

%% ---------------------------------------------------------------
%  Step 3.4: Clean / Target / Attacked / Error map views
% ---------------------------------------------------------------

I_ref    = extractdata(params.desiredAtkImage);
algo_name = upper(params.sar_algo);

xv = bbox(1) + (0:size(I_ref, 2)-1) * dx;
yv = bbox(3) + (0:size(I_ref, 1)-1) * dy;

% (a) Clean image
fig_clean = figure('Color','w');
imagesc(xv, yv, sarImage_clean);
set(gca,'YDir','normal'); axis image;
colormap(gca,'jet'); caxis(clim);
xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
colorbar; title('Clean (True Image)');

% Optional export:
% exportgraphics(fig_clean, sprintf('sar_%s_clean.pdf', algo_name), 'ContentType','vector');
% exportgraphics(fig_clean, sprintf('sar_%s_clean.png', algo_name), 'Resolution', 600);

% (b) Desired target
fig_ref = figure('Color','w');
imagesc(xv, yv, I_ref);
set(gca,'YDir','normal'); axis image;
colormap(gca,'jet'); caxis(clim);
xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
colorbar; title('Target (Desired Attacked)');

% Optional export:
% exportgraphics(fig_ref, sprintf('sar_%s_target_Iref.pdf', algo_name), 'ContentType','vector');
% exportgraphics(fig_ref, sprintf('sar_%s_target_Iref.png', algo_name), 'Resolution', 600);

% (c) Attacked image (I_att)
fig_att = figure('Color','w');
imagesc(xv, yv, I_att);
set(gca,'YDir','normal'); axis image;
colormap(gca,'jet'); caxis(clim);
colorbar; xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
title('Reconstructed (After Attack)');
exportgraphics(fig_att, sprintf('sar_%s_attacked_Iatt.pdf', algo_name), 'ContentType','vector');
exportgraphics(fig_att, sprintf('sar_%s_attacked_Iatt.png', algo_name), 'Resolution', 600);

% (d) Error map |I_att - I_ref|
err_map = abs(I_att - I_ref);
fig_err = figure('Color','w');
imagesc(xv, yv, err_map);
set(gca,'YDir','normal'); axis image;
colormap(gca,'jet'); xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
colorbar; title('|I_{att} - I_{ref}|');

% Optional export:
% exportgraphics(fig_err, sprintf('sar_%s_error_map.pdf', algo_name), 'ContentType','vector');
% exportgraphics(fig_err, sprintf('sar_%s_error_map.png', algo_name), 'Resolution', 600);

%% ---------------------------------------------------------------
%  Step 4: Quantitative Attack Evaluation
% ---------------------------------------------------------------

I_ref = extractdata(params.desiredAtkImage);
fprintf('\n--- Spoofing Attack Performance ---\n');

% 1. MSE
mse_val = mean((I_att(:) - I_ref(:)).^2);
fprintf('Mean Squared Error (MSE): %.6e\n', mse_val);

% 2. RMSE
rmse_val = sqrt(mse_val);
fprintf('Root Mean Squared Error (RMSE): %.6e\n', rmse_val);

% 3. NCC
I_ref_vec = I_ref(:);
I_att_vec = I_att(:);
numerator  = sum(I_ref_vec .* I_att_vec);
denominator = sqrt(sum(I_ref_vec.^2) * sum(I_att_vec.^2));
ncc_val    = numerator / (denominator + eps);
fprintf('Normalized Cross-Correlation (NCC): %.4f\n', ncc_val);

% 4. SSIM
if license('test', 'Image_Toolbox')
    ssim_val = ssim(I_att, I_ref, 'DynamicRange', max(I_ref(:)));
    fprintf('Structural Similarity Index (SSIM): %.4f\n', ssim_val);
else
    fprintf('Structural Similarity Index (SSIM): Skipped (Image Processing Toolbox required).\n');
    ssim_val = NaN;
end

% 5. PSNR
if license('test', 'Image_Toolbox')
    max_val = max(I_ref(:));
    psnr_val = psnr(I_att, I_ref, max_val);
    fprintf('Peak Signal-to-Noise Ratio (PSNR): %.2f dB\n', psnr_val);
else
    fprintf('Peak Signal-to-Noise Ratio (PSNR): Skipped (Image Processing Toolbox required).\n');
end

%% ---------------------------------------------------------------
%  Step 5: Visual Comparison and Spectral Analysis
% ---------------------------------------------------------------

% 5.1 Side-by-side comparison
figure('Name','Attack Evaluation: Image Comparison', ...
       'Units','normalized','Position',[0.05 0.1 0.9 0.45]);

subplot(1,4,1);
imagesc(abs(extractdata(params.trueImage))); axis image off;
title('Clean (True Image)'); colormap jet; colorbar;

subplot(1,4,2);
imagesc(I_ref); axis image off;
title('Target (Desired Attacked)'); colormap jet; colorbar;

subplot(1,4,3);
imagesc(I_att); axis image off;
title('Reconstructed (After Attack)'); colormap jet; colorbar;

subplot(1,4,4);
imagesc(abs(I_att - I_ref)); axis image off;
title('|I_{att} - I_{ref}| (Error Map)');
colormap hot; colorbar;
sgtitle('Spoofing Attack Visual Assessment');

% 5.2 Scatter plot (pixel correlation)
figure('Name','Pixel Correlation', ...
       'Units','normalized','Position',[0.3 0.3 0.4 0.4]);
scatter(I_ref(:), I_att(:), 5, '.'); hold on;

maxVal = max([max(I_ref(:)), max(I_att(:))]);
pad    = 0.05 * maxVal;
plot([0, maxVal+pad], [0, maxVal+pad], 'r--', 'LineWidth', 1.2);

xlabel('Target Intensity I_{ref}');
ylabel('Attacked Intensity I_{att}');
axis equal; grid on;
xlim([0, maxVal+pad]);
ylim([0, maxVal+pad]);
title(sprintf('Pixel Correlation (NCC=%.3f, SSIM=%.3f)', ncc_val, ssim_val));

% 5.3 Error histogram
figure('Name','Error Histogram', ...
       'Units','normalized','Position',[0.3 0.3 0.4 0.4]);
histogram(I_att(:) - I_ref(:), 100);
xlabel('Pixel Error (I_{att} - I_{ref})');
ylabel('Count');
title('Error Distribution');
grid on;

% 5.4 Compact error histogram with fixed x-limits
err = I_att(:) - I_ref(:);
figure('Name','Pixel Error Histogram (Limited Range)', ...
       'Color','w', ...
       'Units','normalized', ...
       'Position',[0.25 0.35 0.50 0.25]);
histogram(err, 80);
grid on;
xlim([-1.5 1.5]);
xlabel('Pixel error  (I_{att} - I_{ref})', 'Interpreter','tex');
ylabel('Number of pixels', 'Interpreter','tex');
set(gca, 'FontSize', 12);

% Optional export:
% exportgraphics(gcf, 'pixel_error_histogram.pdf', 'ContentType','vector');
% exportgraphics(gcf, 'pixel_error_histogram.png', 'Resolution', 600);

% 5.5 Log-spectrum comparison
figure('Name','Log-Spectrum Comparison', ...
       'Units','normalized','Position',[0.05 0.1 0.9 0.4]);
subplot(1,3,1);
imagesc(log10(abs(fftshift(fft2(I_ref))) + 1e-12)); axis image off;
title('Target Spectrum'); colormap jet; colorbar;

subplot(1,3,2);
imagesc(log10(abs(fftshift(fft2(I_att))) + 1e-12)); axis image off;
title('Attacked Spectrum'); colormap jet; colorbar;

subplot(1,3,3);
imagesc(log10(abs(fftshift(fft2(I_att))) + 1e-12) - ...
        log10(abs(fftshift(fft2(I_ref))) + 1e-12));
axis image off; colormap jet; colorbar;
title('Spectral Difference (dB)');
sgtitle('Frequency-Domain Similarity');

% 5.6 Error overlay on target
err_map     = abs(I_att - I_ref);
overlay     = mat2gray(I_ref);
overlay_rgb = ind2rgb(gray2ind(overlay,256), jet(256));
err_overlay = imoverlay(overlay_rgb, ...
                        err_map > 0.1 * max(err_map(:)), [1 0 0]);
figure; imshow(err_overlay);
title('Error Overlay (Red = High Discrepancy)');

% 5.7 3D surface plots: clean / target / attacked / error
[y_idx, x_idx] = size(I_ref);
xv             = bbox(1) + (0:x_idx-1) * dx;
yv             = bbox(3) + (0:y_idx-1) * dy;
[Xg, Yg]       = meshgrid(xv, yv);

error_surface = abs(I_att - I_ref);

ZLim_mag = [0, max([max(trueImage_abs(:)), max(I_ref(:)), max(I_att(:))])];
ZLim_err = [0, max(error_surface(:))];

figure('Name','3D Camouflage Attack Surface Analysis', ...
       'Units','normalized','Position',[0.1 0.1 0.8 0.8]);

view_angle = [30, 45];

subplot(2,2,1);
surf(Xg, Yg, trueImage_abs, 'EdgeColor','none', 'FaceColor','interp');
caxis(ZLim_mag); zlim(ZLim_mag);
colormap(gca,'jet'); colorbar;
title('1. Clean Image (I_{clean})');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Magnitude');
view(view_angle);

subplot(2,2,2);
surf(Xg, Yg, I_ref, 'EdgeColor','none', 'FaceColor','interp');
caxis(ZLim_mag); zlim(ZLim_mag);
colormap(gca,'jet'); colorbar;
title('2. Desired Target (I_{ref})');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Magnitude');
view(view_angle);

subplot(2,2,3);
surf(Xg, Yg, I_att, 'EdgeColor','none', 'FaceColor','interp');
caxis(ZLim_mag); zlim(ZLim_mag);
colormap(gca,'jet'); colorbar;
title('3. Attacked Result (I_{att})');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Magnitude');
view(view_angle);

subplot(2,2,4);
surf(Xg, Yg, error_surface, 'EdgeColor','none', 'FaceColor','interp');
caxis(ZLim_err); zlim(ZLim_err);
colormap(gca,'jet'); colorbar;
title('4. |I_{att} - I_{ref}| (Error Surface)');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Error Magnitude');
view(view_angle);

linkprop([subplot(2,2,1), subplot(2,2,2), subplot(2,2,3), subplot(2,2,4)], ...
         {'CameraPosition','CameraTarget','CameraUpVector'});

%% ---------------------------------------------------------------
%  Optional: PSF-style 2D and 1D plots (commented by default)
% ---------------------------------------------------------------
%{
I_clean = sarImage_clean;
I_atk   = I_att;

[~, idx_max]    = max(I_clean(:));
[row_peak, col_peak] = ind2sub(size(I_clean), idx_max);

half_win_y = 25;
half_win_x = 25;

rows = max(1, row_peak-half_win_y) : min(size(I_clean,1), row_peak+half_win_y);
cols = max(1, col_peak-half_win_x) : min(size(I_clean,2), col_peak+half_win_x);

I_clean_psf = I_clean(rows, cols);
I_atk_psf   = I_atk(rows, cols);

I_clean_psf = I_clean_psf / max(I_clean_psf(:));
I_atk_psf   = I_atk_psf   / max(I_atk_psf(:));

xv_full = bbox(1) + (0:size(I_clean,2)-1) * dx;
yv_full = bbox(3) + (0:size(I_clean,1)-1) * dy;
xv_psf  = xv_full(cols);
yv_psf  = yv_full(rows);

figure('Name','2D PSF: Clean vs Attacked','Color','w','Position',[100 100 900 400]);
subplot(1,2,1);
imagesc(xv_psf, yv_psf, I_clean_psf);
set(gca,'YDir','normal'); axis image;
colormap(gca,'jet'); colorbar; caxis([0 1]);
xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
title('Clean PSF (Normalized)');

subplot(1,2,2);
imagesc(xv_psf, yv_psf, I_atk_psf);
set(gca,'YDir','normal'); axis image;
colormap(gca,'jet'); colorbar; caxis([0 1]);
xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
title('Attacked PSF (Normalized)');

I_clean_line = I_clean(row_peak, :);
I_atk_line   = I_atk(row_peak, :);

eps_lin       = 1e-12;
clean_line_db = 20*log10(I_clean_line / (max(I_clean_line)+eps_lin) + eps_lin);
atk_line_db   = 20*log10(I_atk_line   / (max(I_atk_line)+eps_lin)   + eps_lin);

xv = xv_full;
figure('Name','1D PSF Horizontal Cut','Color','w','Position',[100 100 800 400]);
plot(xv, clean_line_db, 'LineWidth',1.5); hold on;
plot(xv, atk_line_db,   'LineWidth',1.5);
grid on;
xlabel('Horizontal (mm)');
ylabel('Normalized (dB)');
legend({'Clean','Attacked'}, 'Location','SouthWest');
ylim([-60 5]);

% Optional export:
% exportgraphics(gcf, 'psf_horizontal_cut.pdf', 'ContentType','vector');
% exportgraphics(gcf, 'psf_horizontal_cut.png', 'Resolution', 600);
%}

%% ---------------------------------------------------------------
%  Helper Functions (SAR Processing and Visualization)
% ---------------------------------------------------------------

function [loss, gradRe, gradIm] = loss_and_grad(X_v, D, A_re, A_im, params, lambda_L2)
% LOSS_AND_GRAD: MSE loss + L2 regularization and gradients w.r.t. A_re, A_im.
% Supports: params.sar_algo = 'MFA' | 'RMA' | 'BPA' | 'LIA'

    if ~isa(X_v, 'dlarray'), X_v = dlarray(X_v); end
    if ~isa(D,   'dlarray'), D   = dlarray(D);   end
    if nargin < 6 || isempty(lambda_L2), lambda_L2 = 0; end

    A = A_re + 1j * A_im;                 % (Np x 1)
    Y = X_v + D .* A.';                   % (Nsamp x Np)

    Y_cube = reshape(Y, params.Nsamp, params.M, params.N);

    algo = upper(params.sar_algo);
    switch algo
        case 'MFA'
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :));
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end
            [~, ~, atkImage, ~] = dlMFA(sarData, params);

        case 'RMA'
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :));
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end
            [~, ~, atkImage, ~] = dlRMA(sarData, params);

        case 'BPA'
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :));
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end
            if ~isfield(params, 'H_bpa')
                error('params.H_bpa is required for BPA in loss_and_grad.');
            end
            [~, ~, atkImage, ~] = dlBPA(sarData, params, params.H_bpa);

        case 'LIA'
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :));
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end
            if ~isfield(params, 'H_bpa')
                error('params.H_bpa is required for LIA in loss_and_grad.');
            end
            if ~isfield(params, 'py')
                error('params.py is required for LIA in loss_and_grad.');
            end
            % For gradient, reuse BPA operator (faster)
            [~, ~, atkImage, ~] = dlBPA(sarData, params, params.H_bpa);

        otherwise
            error('Unknown params.sar_algo = %s', params.sar_algo);
    end

    loss_im = mean((atkImage - params.desiredAtkImage).^2, 'all');
    reg     = lambda_L2 * mean(abs(A).^2, 'all');
    loss    = loss_im + reg;

    [gradRe, gradIm] = dlgradient(loss, A_re, A_im);
end

%%%%%%%%%%%%%%%%%%%%% LIA (Li & Chen iterative imaging) %%%%%%%%%%%%%%%%%%%%
function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlLIA(sarData, params, H_bpa)
% DLLIA: Lightweight Iterative Imaging Algorithm (LIA) using the
%        same propagation matrix H_bpa as BPA.
%
% Inputs:
%   sarData  : M x N complex slice at the chosen range bin (serpentine corrected)
%   params   : struct with fields:
%              - M, N        : aperture size
%              - A_bpa, B_bpa: image size (A x B)
%              - bbox, dx, dy, z0
%              - py          : index subset of sensor samples (length kk)
%   H_bpa    : (M*N) x (A*B) propagation matrix from dlBPA_H_matrix
%
% Outputs:
%   xRangeT, yRangeT  : spatial axes (mm)
%   trueImage_abs     : RMS-normalized magnitude image (dlarray, 'SS')
%   trueImage_complx  : complex image (B x A)

    % Ensure dlarray types for AD compatibility
    if ~isa(sarData, 'dlarray')
        sarData = dlarray(sarData);
    end
    if ~isa(H_bpa, 'dlarray')
        H_bpa = dlarray(H_bpa);
    end

    M  = params.M;
    N  = params.N;
    A  = params.A_bpa;
    B  = params.B_bpa;
    py = params.py;          % index subset (kk x 1)
    bbox = params.bbox;
    dx   = params.dx;
    dy   = params.dy;

    % --- Vectorize measurements (same ordering as dlBPA_H_matrix) ---
    rd_full = reshape(sarData, [], 1);    % (M*N) x 1
    rd      = rd_full(py);                % kk x 1

    % --- Sub-sampled propagation matrix Hp (kk x BA) ---
    Hp = H_bpa(py, :);                    % kk x (A*B)
    BA = A * B;

    % --- LIA core (myalg == 6 in Li & Chen SPL paper) ---
    di = 0.01;                            % initialization constant
    G  = di * (Hp' * Hp);                 % (BA x BA)
    xd = di * (Hp' * rd);                 % (BA x 1)

    % Iterative image updating based on matrix inversion lemma
    for j = 1:BA
        Gj    = G(:, j);                  % (BA x 1)
        denom = 1 + G(j, j);
        temp  = Gj / denom;               % (BA x 1)

        xd = xd - temp * xd(j);
        G  = G  - temp * G(j, :);
    end

    % ---- EXTRACT DIAGONAL AS COLUMN (dlarray-safe, no broadcasting) ----
    BA     = size(G, 1);
    diagG  = G(1:BA+1:BA*BA);        % 1 x BA (row)
    diagG  = reshape(diagG, [BA, 1]); % BA x 1 (column)

    % Final scaling (element-wise)
    xd = xd ./ diagG;                % stays BA x 1

    % Reshape to B x A and flip horizontally (as in original code)
    xdi = fliplr(reshape(xd, B, A));

    % --- Outputs ---
    trueImage_complx = xdi;

    img_mag = abs(trueImage_complx);
    rms_val = sqrt(mean(img_mag(:).^2) + eps);
    trueImage_abs = dlarray(img_mag ./ rms_val, 'SS');

    % Spatial ranges (mm)
    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
end

%%%%%%%%%%%%%%%%%%%%% MFA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
function matchedFilter = refMF(params)
% REFMF: Generates the 2D Matched Filter coefficient.
    c = physconst('lightspeed');
    x = params.dx * (-(params.nFFTspace-1)/2 : (params.nFFTspace-1)/2) * 1e-3;
    y = (params.dy * (-(params.nFFTspace-1)/2 : (params.nFFTspace-1)/2) * 1e-3).';
    z0_m = params.z0 * 1e-3;
    k = 2 * pi * params.F0 / c;
    
    % Matched Filter: exp(-j * 2 * k * R)
    matchedFilter = exp(-1i * 2 * k * sqrt(bsxfun(@plus, x.^2, y.^2) + z0_m^2));
end

function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlMFA(sarData, params)
% DLMFA: 2D Matched Filter Algorithm for SAR image reconstruction.
    matchedFilter = refMF(params);
    if isa(sarData,'dlarray') && ~isa(matchedFilter,'dlarray')
        matchedFilter = dlarray(matchedFilter);
    end
    
    [yPointM, xPointM] = size(sarData);
    [yPointF, xPointF] = size(matchedFilter);
    
    % 1. Equalize Dimensions with Zero Padding
    % Pad sarData to match matchedFilter size (assuming nFFTspace >= M, N)
    if (xPointF > xPointM)
        pad_x_pre = floor((xPointF - xPointM) / 2);
        pad_x_post = ceil((xPointF - xPointM) / 2);
        sarData = cat(2, zeros(yPointM, pad_x_pre, 'like', sarData), sarData, zeros(yPointM, pad_x_post, 'like', sarData));
        xPointM = xPointF; % Update size
    end
    if (yPointF > yPointM)
        pad_y_pre = floor((yPointF - yPointM) / 2);
        pad_y_post = ceil((yPointF - yPointM) / 2);
        sarData = cat(1, zeros(pad_y_pre, xPointM, 'like', sarData), sarData, zeros(pad_y_post, xPointM, 'like', sarData));
        yPointM = yPointF; % Update size
    end
    % Note: Padding matchedFilter to match sarData is omitted as sarData is padded up to nFFTspace.

    % 2. Run 2D FFT, Multiply, and 2D IFFT (Correlation)
    sarDataFFT = fft(fft(sarData, [], 2), [], 1);
    matchedFilterFFT = fft(fft(matchedFilter, [], 2), [], 1);
    
    trueImage_shifted = ifft(ifft(sarDataFFT .* matchedFilterFFT, [], 2), [], 1);
    
    % 3. Shift and Crop Image
    trueImage = fftshift(trueImage_shifted);
    
    [J, I] = size(trueImage);
    
    % Calculate crop indices based on bounding box
    xij = round(params.bbox(1:2) / params.dx - 0.5 + I/2);
    ykl = round(params.bbox(3:4) / params.dy - 0.5 + J/2);
    
    trueImage_cropped = trueImage(ykl(1):ykl(2), xij(1):xij(2));
    trueImage_cropped = fliplr(trueImage_cropped);
    trueImage_complx = trueImage_cropped;
    
    % RMS-normalize magnitude output
    img_mag = abs(trueImage_cropped);
    rms_val = sqrt(mean(img_mag(:).^2) + eps);
    trueImage_abs = dlarray(img_mag ./ rms_val, 'SS');
    
    % Spatial ranges for plotting
    xRangeT = params.bbox(1) + (0:size(trueImage_abs, 2) - 1) * params.dx;
    yRangeT = params.bbox(3) + (0:size(trueImage_abs, 1) - 1) * params.dy;
end


%%%%%%%%%%%%%%%%%%%%% BPA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlBPA(sarData, params, H)
% DLBPA_WRAPPER: Back-Projection Algorithm (BPA)

    % --- 1. Extract Parameters and Data ---
    %H = params.H_bpa;       % Pre-computed H matrix (dlarray)
    A = params.A_bpa;       % Image horizontal pixels
    B = params.B_bpa;       % Image vertical pixels
    %M = params.M;           % Aperture horizontal points
    %N = params.N;           % Aperture vertical points
    
    % Ensure input data is a dlarray
    if ~isa(sarData, 'dlarray')
        sarData = dlarray(sarData);
    end
    
    % Vectorize measurements: y is (M*N) x 1 (the 'rd(py)' equivalent)
    % We assume the sarData (M x N) matches the construction order of H.
    y = reshape(sarData, [], 1);

    % --- 2. Direct Back-Projection (Matched Filter) ---
    % Operation: xd = H' * y 
    % H' is the conjugate transpose (the correct Back-Projection operator)
    xd = H' * y;
    
    % Reshape: xd is (B*A) x 1 --> B x A image matrix
    xdi = reshape(xd, B, A);
    
   
    % Orientation Correction: fliplr(reshape(xd, B, A))
    trueImage_cropped = fliplr(xdi); 
    trueImage_complx = trueImage_cropped; % Complex image output
    %trueImage_complx = xdi; % Complex image output
    
    % RMS-normalize magnitude output
    img_mag = abs(trueImage_complx);
    rms_val = sqrt(mean(img_mag(:).^2) + eps);
    trueImage_abs = dlarray(img_mag ./ rms_val, 'SS'); % Magnitude output as dlarray

    % --- 5. Spatial Ranges (for visualization helpers) ---
    xRangeT = params.bbox(1) + (0:size(trueImage_abs, 2) - 1) * params.dx;
    yRangeT = params.bbox(3) + (0:size(trueImage_abs, 1) - 1) * params.dy;
end


function H = dlBPA_H_matrix(params)
% DLBPA_H_MATRIX: Builds the propagation matrix H (Measurements x Pixels) 
% for the Back-Projection Algorithm (BPA) using only the params structure.

    % --- 1. Extract Dimensions and Parameters ---
    M = params.M;       % Aperture horizontal points
    N = params.N;       % Aperture vertical points
    A = params.A_bpa;   % Image horizontal pixels
    B = params.B_bpa;   % Image vertical pixels
    
    c0 = physconst('lightspeed');
    F0 = params.F0;     % Note: F0 is still needed here for the propagation constant
    z0_mm = params.z0;  % Target range in mm
    dx = params.dx;     % Horizontal step in mm
    dy = params.dy;     % Vertical step in mm
    bbox = params.bbox; % Bounding box in mm

    % --- 2. Convert to Meters (m) ---
    z0_m = z0_mm * 1e-3; 
    dxm = dx * 1e-3; 
    dym = dy * 1e-3; 
    bbox_m = bbox * 1e-3;
    
    % Constants
    % Propagation constant: j * 2 * k = j * 2 * (2*pi*F0/c0)
    k   = 2*pi*F0/c0;
    cst = 1i * 2 * k; 
    %cst = sqrt(-1) * 2 * pi * F0 * 2 / c0; 
    z2 = (z0_m)^2; % z^2 term for distance calculation
    
    % --- 3. Define Image Pixel Coordinates (P_x, P_y) ---
    wh1 = linspace(bbox_m(1), bbox_m(2), A); % Horizontal pixels (m)
    wh2 = linspace(bbox_m(3), bbox_m(4), B); % Vertical pixels (m)
    
    % --- 4. Define Sensor/Aperture Coordinates (S_x, S_y) ---
    [ix_vec, iy_vec] = meshgrid(0:M-1, 0:N-1);
    
    % Sensor positions relative to center (0, 0)
    sx = (ix_vec(:) + 0.5 - M/2) * dxm; % Sensor X coordinates (m)
    sy = (iy_vec(:) + 0.5 - N/2) * dym; % Sensor Y coordinates (m)
    
    % --- 5. Build H Matrix (Iterative Distance Calculation) ---
    NM = M * N; % Total number of measurements (rows of H)
    BA = A * B; % Total number of pixels (columns of H)
    
    H_val = complex(zeros(NM, BA));
    
    fprintf('    Building H matrix (%d x %d)...', NM, BA);
    tic;
    
    % Measurement loop (i) iterates through all NM positions.
    for i = 1:NM 
        
        % --- Sensor Index Calculation (Adapted from sparse code, using full indices) ---
        % Since we use ALL M*N measurements, we map the linear index 'i' to (ix, iy).
        % N is the size of the inner dimension (vertical, iy), M is the size of the outer dim (horizontal, ix).
        % NOTE: This assumes COLUMN-MAJOR vectorization (reading down N, then across M).
        iy = mod(i-1, N); % Vertical index (0 to N-1)
        ix = (i-1-iy)/N;  % Horizontal index (0 to M-1)
        
        % Sensor Coordinates (S_x, S_y) based on explicit indices
        sx_i = (ix + 0.5 - M/2) * dxm; % Sensor X coordinate (m)
        sy_i = (iy + 0.5 - N/2) * dym; % Sensor Y coordinate (m)
    
        % Iterate over all image pixels (columns, j)
        for j = 1:BA 
            % Get pixel indices (jx, jy)
            jy = mod(j-1, B); jx = (j-1-jy)/B; 
            
            % Pixel coordinates (P_x, P_y)
            px = wh1(jx+1);
            py = wh2(jy+1);
            
            % Distance squared: R^2 = (S_x - P_x)^2 + (S_y - P_y)^2 + z^2
            dist2 = (sx_i - px)^2 + (sy_i - py)^2 + z2;
            
            % Propagation channel element: H_ij = exp(j * 2 * k * R)
            H_val(i, j) = exp(cst * sqrt(dist2));
        end
    end
    
    % --- 6. Finalize and Convert ---
    fprintf([' ' num2str(toc, '%.3f') ' sec\n']);
    
    % Convert the complex matrix to dlarray
    H = dlarray(H_val);
end

%%%%%%%%%%%%%%%%%%%%% RMA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlRMA(sarData, params)
% DLRMA: 2D Range Migration Algorithm for SAR image reconstruction using dlarray.
% This function is a dlarray-compatible version of the imaging_2DRMA core logic.

    % Extract parameters from struct
    nFFTspace = params.nFFTspace;
    z0_mm = params.z0; 
    dx = params.dx;
    dy = params.dy;
    bbox = params.bbox;
    F0 = params.F0;

    % Ensure sarData is a dlarray if not already
    isDlArray = isa(sarData, 'dlarray');
    if ~isDlArray
        sarData = dlarray(sarData);
    end

    % 1. Spatial Frequency Domain Setup (kX, kY)
    c = physconst('lightspeed');
    k = 2 * pi * F0 / c;
    
    % kX and kY domains
    wSx = 2 * pi / (dx * 1e-3);
    kX = linspace(-(wSx / 2), (wSx / 2), nFFTspace);
    wSy = 2 * pi / (dy * 1e-3);
    kY = (linspace(-(wSy / 2), (wSy / 2), nFFTspace)).';
    
    % Wave number in z-direction (K_z)
    % K = sqrt((2*k)^2 - kX^2 - kY^2)
    K = sqrt((2*k).^2 - bsxfun(@plus, kX.^2, kY.^2));
    
    % Convert K to dlarray for processing
    if ~isDlArray
        K = dlarray(K);
    end
    
    % 2. Phase Factor (Range Migration Correction and Focus)
    % phaseFactor0 = exp(-i * z0 * K_z)
    phaseFactor0 = exp(-1i * z0_mm * K);
    
    % Set imaginary/evanescent components to zero (where (kX^2 + kY^2) > (2k)^2)
    % For dlarray, use extractdata/gather for complex conditional indexing
    %K_mag_sq = bsxfun(@plus, extractdata(kX).^2, extractdata(kY).^2);
    K_mag_sq = bsxfun(@plus, kX.^2, kY.^2);
    evanescent_mask = K_mag_sq > (2 * k)^2;

    %phaseFactor0 = extractdata(phaseFactor0);
    phaseFactor0(evanescent_mask) = 0;
    phaseFactor0 = dlarray(phaseFactor0);

    % The RMA phase factor for frequency domain data is typically exp(-i*z*K_z)
    % Your original code uses K .* phaseFactor0; we stick to K_z (K) for now.
    %phaseFactor = phaseFactor0;
    phaseFactor = K .* phaseFactor0;
    phaseFactor = fftshift(fftshift(phaseFactor, 1), 2);
    
    % 3. Data Padding and Frequency Domain Processing
    [yPointM, xPointM] = size(sarData);
    [yPointF, xPointF] = size(phaseFactor);

    % Equalize Dimensions of sarData and Phase Factor with Zero Padding
    % Note: Padarray is not natively dlarray compatible, use cat with zeros.
    
    % Pad X (Horizontal)
    if (xPointF > xPointM)
        pad_x_pre = floor((xPointF - xPointM) / 2);
        pad_x_post = ceil((xPointF - xPointM) / 2);
        % Use 'like' to maintain dlarray status if sarData is dlarray
        sarData = cat(2, zeros(yPointM, pad_x_pre, 'like', sarData), sarData, zeros(yPointM, pad_x_post, 'like', sarData));
    elseif (xPointM > xPointF)
        % Pad Phase Factor (shouldn't happen if nFFTspace is >= M, N)
        pad_x_pre = floor((xPointM - xPointF) / 2);
        pad_x_post = ceil((xPointM - xPointF) / 2);
        phaseFactor = cat(2, zeros(yPointF, pad_x_pre, 'like', phaseFactor), phaseFactor, zeros(yPointF, pad_x_post, 'like', phaseFactor));
    end
    
    % Pad Y (Vertical)
    if (yPointF > yPointM)
        pad_y_pre = floor((yPointF - yPointM) / 2);
        pad_y_post = ceil((yPointF - yPointM) / 2);
        sarData = cat(1, zeros(pad_y_pre, size(sarData, 2), 'like', sarData), sarData, zeros(pad_y_post, size(sarData, 2), 'like', sarData));
    elseif (yPointM > yPointF)
        pad_y_pre = floor((yPointM - yPointF) / 2);
        pad_y_post = ceil((yPointM - yPointF) / 2);
        phaseFactor = cat(1, zeros(pad_y_pre, size(phaseFactor, 2), 'like', phaseFactor), phaseFactor, zeros(pad_y_post, size(phaseFactor, 2), 'like', phaseFactor));
    end
    
    % 4. 2D IFFT (Image Formation)
    %sarDataFFT = fft2(sarData, nFFTspace, nFFTspace);
    sarDataFFT = fft(fft(sarData, [], 2), [], 1);


    % Element-wise multiplication (Multiplication is gradient-aware)
    %trueImage_shifted = ifft2(sarDataFFT .* phaseFactor);
    trueImage= ifft(ifft(sarDataFFT .* phaseFactor, [], 2), [], 1);
    
    % Shift to center the image (like dlMFA)
    %trueImage = fftshift(trueImage_shifted);
    
    % 5. Crop and Normalize
    [J, I] = size(trueImage);

    % Calculate crop indices based on bounding box
    xij = round(bbox(1:2) / dx - 0.5 + I/2);
    ykl = round(bbox(3:4) / dy - 0.5 + J/2);

    trueImage_cropped = trueImage(ykl(1):ykl(2), xij(1):xij(2));

    trueImage_cropped = fliplr(trueImage_cropped);

    trueImage_complx = trueImage_cropped;
    
    % RMS-normalize magnitude output
    img_mag = abs(trueImage_cropped);
    rms_val = sqrt(mean(img_mag(:).^2) + eps);
    trueImage_abs = img_mag ./ rms_val;
    
    % Set spatial ranges for plotting (consistent with dlMFA)
    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
end


function plot_sar_bpa(I_plot, bbox, dx, dy, plotTitle)
    % PLOT_SAR_BPA: Plots the image, using linspace to define axes based on bbox.
    %
    % Usage:
    %   plot_sar_bpa(I, bbox, dx, dy);              % no title
    %   plot_sar_bpa(I, bbox, dx, dy, 'BPA image'); % with title

    % Check if title is provided
    if nargin < 5 || isempty(plotTitle)
        useTitle = false;
    else
        useTitle = true;
    end

    [B, A] = size(I_plot);

    xv = linspace(bbox(1), bbox(2), A); % Horizontal axes (mm)
    yv = linspace(bbox(3), bbox(4), B); % Vertical axes (mm)

    figure;
    mesh(xv, yv, I_plot, 'FaceColor', 'interp', 'LineStyle', 'none');
    view(2);
    colormap('jet');
    axis equal tight;

    xlabel('Horizontal (mm)');
    ylabel('Vertical (mm)');

    if useTitle
        title(plotTitle);
    end

    xlim([bbox(1) bbox(2)]);
    ylim([bbox(3) bbox(4)]);
end

function plot_sar(sarImage, bbox, dx, dy, plotTitle)
% Helper for standard mesh plot
%
% Usage:
%   plot_sar(img, bbox, dx, dy);             % no title
%   plot_sar(img, bbox, dx, dy, 'My Title'); % with title

    % Check if title is provided
    if nargin < 5 || isempty(plotTitle)
        useTitle = false;
    else
        useTitle = true;
    end

    xv = bbox(1) + (0:size(sarImage, 2) - 1) * dx;
    yv = bbox(3) + (0:size(sarImage, 1) - 1) * dy;

    mesh(xv, yv, sarImage, 'FaceColor', 'interp', 'LineStyle', 'none');
    view(2);
    colormap('jet');
    axis equal tight;
    xlabel('Horizontal (mm)');
    ylabel('Vertical (mm)');

    if useTitle
        title(plotTitle);
    end

    xlim([bbox(1) bbox(2)]);
    ylim([bbox(3) bbox(4)]);
end

