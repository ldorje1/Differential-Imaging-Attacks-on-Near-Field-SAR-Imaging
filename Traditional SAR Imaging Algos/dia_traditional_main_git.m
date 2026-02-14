%% ========================================================================
%  Differential Imaging Attack (DIA) for Near-Field SAR/mmWave Imaging
%
%  End-to-end differentiable attack that optimizes complex per-aperture gains
%  A to drive a chosen SAR reconstructor (MFA/RMA/BPA/LIA) toward a target
%  image. The script:
%    1) Loads victim FMCW SAR data, gates a range bin, and reconstructs a
%       clean image.
%    2) Builds a target image by shuffling gated measurements and re-imaging
%       (BPA target used in LIA mode for consistent scaling).
%    3) Samples attack waveforms from X_aa, frequency-aligns them to the
%       victim range bin, and RMS-matches to the victim measurements.
%    4) Runs gradient-based optimization on A with optional |A|<=Amax
%       projection and L2 regularization.
%    5) Reconstructs the final attacked image and reports MSE/NCC/SSIM/PSNR
%       plus signal-domain power ratio Pa/Pr.
% ========================================================================

clc; clear; close all;

%% ---------------------------------------------------------------
%  User selection: SAR imaging algorithm
% ---------------------------------------------------------------
dataDir  = fullfile(pwd, 'data');          % folder holding .mat files
sar_algo = 'MFA';                          % Options: 'MFA', 'RMA', 'BPA', 'LIA'

%% ---------------------------------------------------------------
%  Step 2: Load Victim SAR Data and Setup Imaging Parameters
% ---------------------------------------------------------------
sarRawData = load(fullfile(dataDir, 'rawSAR.mat')).adcDataCube;   % Nsamp x M x N
[Nsamp, M, N] = size(sarRawData);

X_v = reshape(sarRawData, Nsamp, M * N);   % Nsamp x Np (vectorize aperture)
Np  = M * N;

% SAR system parameters
c0 = physconst('lightspeed');
F0 = 77e9;          % start frequency (Hz)
FS = 5000e3;        % sampling rate (samples/s)
Ts = 1 / FS;        % sampling period (s)
K0 = 70.295e12;     % chirp slope (Hz/s)
tI = 4.5225e-10;    % instrument delay (s)

nFFTtime  = 1024;   % range-FFT points
nFFTspace = 1024;   % spatial FFT points (MFA/RMA)

% ---------------------------------------------------------------
%  Forward imaging model, generate clean and target image
% ---------------------------------------------------------------
switch upper(sar_algo)

    case 'MFA'
        dx   = 1;                         % horizontal step (mm)
        dy   = 1;                         % vertical step (mm)
        bbox = [-200 200 -200 200];       % [xmin xmax ymin ymax] in mm
        z0   = 185;                       % target range (mm)

        % Range-bin index (MFA path) derived from FMCW delay model
        k0_range_bin = round(K0 / FS * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        % Range FFT along fast-time, then gate a single range bin
        rawDataFFT = fft(sarRawData, nFFTtime);
        sarData    = squeeze(rawDataFFT(k0_range_bin + 1, :, :)); % M x N

        % Serpentine correction (flip every other scan row)
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % Package parameters used by the imaging operator
        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        % Clean image (reconstruct from correct sarData slice)
        [~, ~, clean_img, ~] = dlMFA(sarData, params);
        clean_img = extractdata(clean_img);

        % Target image (structured random): shuffle sarData entries, then re-image
        [rows, cols] = size(sarData); rng(42);                              % fixed seed for reproducibility
        sarData_shuffled = reshape(sarData(randperm(rows * cols)), rows, cols);
        [~, ~, target_img, ~] = dlMFA(sarData_shuffled, params);
        target_img = extractdata(target_img);

    case 'RMA'
        % Reorder cube to match your RMA implementation's expected layout
        Echo = permute(sarRawData, [3, 2, 1]);   % [samples, vertical, horizontal] -> [horizontal, vertical, samples]
        dx   = 1;
        dy   = 1;
        bbox = [-200 200 -200 200];

        % RMA uses num_sample FFT points along the sample dimension
        Nx         = 200;
        Nz         = 200;
        num_sample = size(Echo, 3);
        nFFTtime   = num_sample;
        rawDataFFT = fft(Echo, nFFTtime, 3);

        % Application-specific selected bin (you treat this as the gate index)
        ID_select    = 6;
        k0_range_bin = ID_select;

        % Gate the chosen bin and transpose to Nz x Nx style used below
        sarData = squeeze(rawDataFFT(:, :, ID_select)).';
        z0      = c0 / 2 * (ID_select / (K0 * (1 / FS) * nFFTtime) - tI);

        % Serpentine correction (flip every other scan row)
        for ii = 2:2:Nz
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % Package parameters used by the imaging operator
        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        % Clean image
        [~, ~, clean_img, ~] = dlRMA(sarData, params);
        clean_img = extractdata(clean_img);

        % Target image (structured random): shuffle sarData entries, then re-image
        [rows, cols] = size(sarData); rng(42);
        sarData_shuffled = reshape(sarData(randperm(rows * cols)), rows, cols);
        [~, ~, target_img, ~] = dlRMA(sarData_shuffled, params);
        target_img = extractdata(target_img);

    case 'BPA'
        dx   = 1;
        dy   = 1;
        bbox = [-200 200 -200 200];
        z0   = 185;

        % Range FFT along fast-time, then gate a single range bin
        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * Ts * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);
        sarData      = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        % Serpentine correction (flip every other scan row)
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % Image grid size used by BPA
        A = 50;   % horizontal pixels
        B = 50;   % vertical pixels

        % Package parameters for BPA and its H matrix
        params = struct('z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'Nsamp', Nsamp, 'nFFTtime', nFFTtime, 'N', N, 'M', M, ...
                        'A_bpa', A, 'B_bpa', B, 'F0', F0, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        % Precompute propagation matrix for BPA
        H_bpa        = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        % Clean image
        [~, ~, clean_img, ~] = dlBPA(sarData, params, H_bpa);
        clean_img = extractdata(clean_img);

        % Target image (structured random): shuffle sarData entries, then re-image
        [rows, cols] = size(sarData); rng(42);
        sarData_shuffled = reshape(sarData(randperm(rows * cols)), rows, cols);
        [~, ~, target_img, ~] = dlBPA(sarData_shuffled, params, H_bpa);
        target_img = extractdata(target_img);

    case 'LIA'
        dx   = 1;
        dy   = 1;
        bbox = [-200 200 -200 200];
        z0   = 185;

        % Range FFT along fast-time, then gate a single range bin
        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * Ts * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);
        sarData      = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        % Serpentine correction (flip every other scan row)
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % Image grid size used by LIA (and BPA H matrix)
        A = 50;
        B = 50;

        % Package parameters for LIA and its H matrix
        params = struct('z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'Nsamp', Nsamp, 'nFFTtime', nFFTtime, 'N', N, 'M', M, ...
                        'A_bpa', A, 'B_bpa', B, 'F0', F0, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        % Precompute propagation matrix (reused by LIA)
        H_bpa        = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        % Random subset indices for LIA (measurement subsampling)
        NM       = M * N;
        kk       = min(40000, NM);
        rng(1000);
        params.py = sort(randperm(NM, kk));

        % Clean image
        [~, ~, clean_img, ~] = dlLIA(sarData, params, H_bpa);
        clean_img = extractdata(clean_img);

        [~, ~, clean_img_2, ~] = dlBPA(sarData, params, H_bpa);
        clean_img_2 = extractdata(clean_img_2);
        %global_scale_lia = max(abs(clean_img_2(:))) + 1e-12;

        % Target image (structured random): shuffle sarData entries, then re-image
        [rows, cols] = size(sarData); rng(42);
        sarData_shuffled = reshape(sarData(randperm(rows * cols)), rows, cols);
        [~, ~, target_img, ~] = dlBPA(sarData_shuffled, params, H_bpa);
        target_img = extractdata(target_img);

    otherwise
        error('Invalid SAR algorithm selection.');
end

% ---------------------------------------------------------------
% Quick visualization: clean vs shuffled-target
% ---------------------------------------------------------------
figure();
subplot(1, 2, 1);
imagesc(clean_img); axis image off; colormap gray; colorbar;
title(sprintf('Clean Image (%s)', upper(sar_algo)));

subplot(1, 2, 2);
imagesc(target_img); axis image off; colormap gray; colorbar;
title(sprintf('Target Image (%s) - Shuffled', upper(sar_algo)));

%% ---------------------------------------------------------------
% Normalization + store dlarray versions in params
% ---------------------------------------------------------------
if strcmpi(sar_algo, 'LIA')
    % LIA scale (for clean visualization / clean reference)
    global_scale_lia = max(abs(clean_img(:)))   + 1e-12;

    % BPA scale (for target + attack objective consistency)
    global_scale_bpa = max(abs(clean_img_2(:))) + 1e-12;

    params.global_scale_lia = global_scale_lia;
    params.global_scale_bpa = global_scale_bpa;

    % Normalize clean by LIA-scale
    clean_img  = clean_img / global_scale_lia;

    % Normalize target by BPA-scale
    target_img = target_img / global_scale_bpa;

else
    global_scale = max(abs(clean_img(:))) + 1e-12;
    params.global_scale = global_scale;
    %params.global_scale_bpa = global_scale;

    clean_img  = clean_img  / global_scale;
    target_img = target_img / global_scale;
end

% Store dlarray versions for optimization/loss
params.clean_img  = dlarray(clean_img,  "SS");
params.target_img = dlarray(target_img, "SS");

%% ---------------------------------------------------------------
%  Step 3: Build Attack Waveforms and Optimize Complex Gains A
% ---------------------------------------------------------------

% Step 1: Load attack signal pool (X_aa)
temp_x_aa = load(fullfile(dataDir, "X_aa.mat"));      % loads a .mat file that contains X_aa
X_aa      = temp_x_aa.X_aa;                           % Nsamp x (pool_size) complex waveforms

% ---------------------------------------------------------------
% 3.1 Select and frequency-align attack waveforms
% ---------------------------------------------------------------
targetK = 40000;                                      % number of waveforms sampled from pool
rng(0);                                               % fixed seed for reproducible sampling

%sample_idx = randi(p * p, [1, targetK]);             % (old) alternative indexing if pool size were p*p
sample_idx = randi(size(X_aa, 2), [1, targetK]);      % choose targetK random columns from pool [1..16384]
X_a_pool   = X_aa(:, sample_idx);                     % Nsamp x targetK candidate waveforms

sel_idx = randi(size(X_a_pool, 2), [1, Np]);          % pick one waveform index per aperture position
X_a     = X_a_pool(:, sel_idx);                       % Nsamp x Np (one waveform per aperture)

% ---------------------------------------------------------------
% Per-column frequency shift to victim range bin
% ---------------------------------------------------------------
Xspec   = fft(X_a, nFFTtime, 1);                      % FFT along time for each column (Nsamp x Np)
[~, b0] = max(abs(Xspec), [], 1);                     % peak FFT-bin index for each column
f0      = (b0 - 1) * FS / nFFTtime;                   % estimated peak frequency per column (Hz)
f_tgt   = (params.k0_range_bin) * FS / nFFTtime;      % desired target frequency (Hz) corresponding to gate bin
Delta   = f0 - f_tgt;                                 % per-column frequency offset to remove

t = (0:Nsamp-1).' / FS;                               % time axis (Nsamp x 1)
P = exp(-1j * 2*pi * (t * Delta));                    % Nsamp x Np complex exponential (column-wise shift)
D = P .* X_a;                                         % Nsamp x Np shifted waveforms (aligned to victim bin)

% ---------------------------------------------------------------
% Scale D to match victim RMS per column
% ---------------------------------------------------------------
colrms = @(X) sqrt(mean(abs(X).^2, 1));               % RMS magnitude per column
scale  = (colrms(X_v) + eps) ./ (colrms(D) + eps);    % per-column gain to match victim RMS
D      = D .* scale;                                 % RMS-matched injection dictionary

%% ---------------------------------------------------------------
% DIA Optimization
% ---------------------------------------------------------------

% 3.2 Optimization setup (algorithm-specific learning rates)
switch upper(sar_algo)
    case 'MFA'
        maxIter   = 300;                              % number of gradient iterations
        lr_re     = 1e4;                              % step size for real part of A
        lr_im     = 1e4;                              % step size for imag part of A
        lambda_L2 = 1e-5;                             % L2 regularization weight on A
    case 'RMA'
        maxIter   = 300;
        lr_re     = 1e4;
        lr_im     = 1e4;
        lambda_L2 = 1e-5;
    case 'BPA'
        maxIter   = 300;
        lr_re     = 1e4;
        lr_im     = 1e4;
        lambda_L2 = 1e-5;
    case 'LIA'
        maxIter   = 300;
        lr_re     = 1e4;
        lr_im     = 1e4;
        lambda_L2 = 1e-5;
    otherwise
        error('Invalid SAR algorithm selection.');
end

use_projection = true;                                % if true, enforce |A| <= Amax each iteration
Amax           = 2;                                   % amplitude cap used by projection (if enabled)

% Initialize complex gains A = A_re + j*A_im (Np x 1)
A_re = dlarray(1e-3 * randn(Np, 1, 'double'));        % small random initialization (real)
A_im = dlarray(1e-3 * randn(Np, 1, 'double'));        % small random initialization (imag)

% History buffers for logging/debug
loss_hist  = zeros(maxIter, 1);                       % scalar loss per iteration
meanA_hist = zeros(maxIter, 1);                       % mean |A| per iteration
maxA_hist  = zeros(maxIter, 1);                       % max  |A| per iteration

% Console header
fprintf(['\nOptimizing complex gain A for all locations (Np=%d) with SAR algorithm: %s ' ...
         '| lr_re=%.3g, lr_im=%.3g | lambda_L2=%.3g | proj=%d\n'], ...
        Np, params.sar_algo, lr_re, lr_im, lambda_L2, use_projection);

% ---------------------------------------------------------------
% Main optimization loop (gradient descent on A_re, A_im)
% ---------------------------------------------------------------
for iter = 1:maxIter

    % Compute loss + gradients w.r.t. A_re and A_im through the full pipeline
    [loss, gRe, gIm, atkImage] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);

    % Store scalar loss
    lossVal         = double(gather(extractdata(loss)));
    loss_hist(iter) = lossVal;

    % Gradient steps (separate step sizes for real/imag)
    A_re = A_re - lr_re * gRe;
    A_im = A_im - lr_im * gIm;

    % Optional projection: enforce magnitude constraint |A| <= Amax
    if use_projection
        A_num = extractdata(A_re) + 1j * extractdata(A_im);         % numeric complex A
        mags  = abs(A_num);                                         % |A|
        over  = mags > Amax;                                        % constraint violations
        if any(over)
            scale_proj       = ones(size(mags));                    % multiplicative correction
            scale_proj(over) = Amax ./ mags(over);                  % shrink those above Amax
            A_num            = A_num .* scale_proj;                 % project onto ball
            A_re             = dlarray(real(A_num));                % write back to dlarray
            A_im             = dlarray(imag(A_num));
        end
    end

    % Track mean/max |A| for monitoring
    A_now            = extractdata(A_re) + 1j * extractdata(A_im);
    meanA_hist(iter) = mean(abs(A_now));
    maxA_hist(iter)  = max(abs(A_now));

    % -----------------------------------------------------------
    % Logging block (prints at iter=1, every 100 iters, and final)
    % -----------------------------------------------------------
    if mod(iter, 10) == 0 || iter == 1 || iter == maxIter

        % A_log: current A in numeric form (post-projection if enabled)
        A_log = extractdata(A_re) + 1j * extractdata(A_im);

        % Summary stats of A
        meanA = mean(abs(A_log), 'all');
        maxA  = max(abs(A_log),  [], 'all');

        % MSE between attacked image and target/clean (atkImage is a dlarray)
        mse_AT = double(gather(extractdata(mean((atkImage - params.target_img).^2, 'all'))));
        mse_CA = double(gather(extractdata(mean((atkImage - params.clean_img ).^2, 'all'))));

        % Attack-to-victim power ratio Pa/Pr in the signal domain (Frobenius norm squared)
        delta_now = D .* (A_log.');                                % Nsamp x Np injected signal
        PaPr      = (norm(delta_now, 'fro') / (norm(X_v, 'fro') + 1e-12))^2;

        % Gradient magnitude monitor (max absolute component across real/imag)
        gRe_num = extractdata(gRe);
        gIm_num = extractdata(gIm);
        Gnow    = max([max(abs(gRe_num), [], 'all'), max(abs(gIm_num), [], 'all')]);

        % Print log line
        fprintf(['Iter %04d/%04d | Loss=%.4e, G=%.6e | ' ...
                 'MSE(A,T)=%.4e, MSE(C,A)=%.4e | ' ...
                 'E|A|=%.4e, max|A|=%.4e, Pa/Pr=%.4e\n'], ...
                iter, maxIter, lossVal, Gnow, mse_AT, mse_CA, meanA, maxA, PaPr);
    end
end
fprintf('-----------------------------\n');

%% ---------------------------------------------------------------
% 4 Evaluations
% ---------------------------------------------------------------

% ---------------------------------------------------------------
% Reconstruct attacked image with the optimized A
% ---------------------------------------------------------------
A_opt  = extractdata(A_re) + 1j * extractdata(A_im);               % final complex gains (Np x 1)
Y_opt  = X_v + D .* (A_opt.');                                     % attacked measurements (Nsamp x Np)
Y_cube = reshape(Y_opt, Nsamp, M, N);                              % back to Nsamp x M x N cube

% Range FFT and gate same bin used for clean/target generation
rawDataFFT_att = fft(Y_cube, nFFTtime);
sarData_att    = squeeze(rawDataFFT_att(k0_range_bin + 1, :, :));  % M x N

% Serpentine correction for attacked slice
for ii = 2:2:size(sarData_att, 1)
    sarData_att(ii, :) = fliplr(sarData_att(ii, :));
end

% Reconstruct attacked image using the chosen imaging algorithm
switch upper(params.sar_algo)
    case 'MFA'
        [~, ~, atkImage_abs, ~] = dlMFA(sarData_att, params);
        adv_img = gather(extractdata(atkImage_abs));

    case 'RMA'
        [~, ~, atkImage_abs, ~] = dlRMA(sarData_att, params);
        adv_img = gather(extractdata(atkImage_abs));               % (kept as-is)

    case 'BPA'
        [~, ~, atkImage_abs, ~] = dlBPA(sarData_att, params, params.H_bpa);
        adv_img = gather(extractdata(atkImage_abs));

    case 'LIA'
        [~, ~, atkImage_abs, ~] = dlLIA(sarData_att, params, params.H_bpa);
        adv_img = gather(extractdata(atkImage_abs));

    otherwise
        error('Invalid SAR algorithm selection for final reconstruction.');
end

%% ---------------------------------------------------------------
% Final metric evaluation (image domain)
% ---------------------------------------------------------------
if strcmpi(params.sar_algo, 'LIA')
    adv_img = adv_img / params.global_scale_bpa;                   % match target domain (BPA-scale)
else
    adv_img = adv_img / params.global_scale;
end

%%
% MSE(A,T): attacked vs target
mse_val = mean((adv_img(:) - target_img(:)).^2);

% MSE(C,A): attacked vs clean
mse_CA  = mean((adv_img(:) - clean_img(:)).^2);

% NCC between attacked and target (dot-product normalized)
num     = sum(adv_img(:) .* target_img(:));
den     = sqrt(sum(adv_img(:).^2) * sum(target_img(:).^2)) + 1e-12;
ncc_val = num / den;

% SSIM / PSNR (DynamicRange derived from target)
data_range = (max(target_img(:)) - min(target_img(:))) + 1e-12;

if exist('ssim', 'file') == 2
    ssim_val = ssim(adv_img, target_img, 'DynamicRange', data_range);
else
    ssim_val = NaN;
end

if exist('psnr', 'file') == 2
    psnr_val = psnr(adv_img, target_img, data_range);
else
    psnr_val = 10 * log10((data_range^2) / (mse_val + 1e-12));
end

% Print metric summary
fprintf('\n--- %s SR-level attack metrics ---\n', upper(params.sar_algo));
fprintf('MSE(A,T)   : %.4e \n', mse_val);
fprintf('MSE(C,A)   : %.4e\n', mse_CA);
fprintf('NCC        : %.4f\n', ncc_val);
fprintf('SSIM       : %.4f\n', ssim_val);
fprintf('PSNR       : %.2f dB\n', psnr_val);

% ---------------------------------------------------------------
% Visualization: clean, target, attacked, and difference
% ---------------------------------------------------------------
figure();

subplot(2, 2, 1);
imagesc(clean_img); axis image off; colormap gray; colorbar;
title(sprintf('Clean %s output image', upper(params.sar_algo)));

subplot(2, 2, 2);
imagesc(target_img); axis image off; colormap gray; colorbar;
title('Target image (desired attacked)');

subplot(2, 2, 3);
imagesc(adv_img); axis image off; colormap gray; colorbar;
title('Adversarial image (attacked)');

subplot(2, 2, 4);
imagesc(adv_img - clean_img); axis image off; colormap gray; colorbar;
title('Diff (A-C)');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%   FUNCTIONS  %%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%% Gradient Calculation %%%%%%%%%%%%%%%%%%%%
function [loss, gradRe, gradIm, atkImage] = loss_and_grad(X_v, D, A_re, A_im, params, lambda_L2)
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

    if strcmpi(params.sar_algo, 'LIA')
        atkImage = atkImage / params.global_scale_bpa;   % match target domain (BPA-scale)
    else
        atkImage = atkImage / params.global_scale;
    end

    loss_im = mean((atkImage - params.target_img).^2, 'all');
    reg     = lambda_L2 * mean(abs(A).^2, 'all');
    loss    = loss_im + reg;

    [gradRe, gradIm] = dlgradient(loss, A_re, A_im);
end

%% %%%%%%%%%%%%%%%%%%%%% LIA (Li & Chen iterative imaging) %%%%%%%%%%%%%%%%%%%%
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

    M    = params.M;
    N    = params.N;
    A    = params.A_bpa;
    B    = params.B_bpa;
    py   = params.py;          % index subset (kk x 1)
    bbox = params.bbox;
    dx   = params.dx;
    dy   = params.dy;

    % --- Vectorize measurements (same ordering as dlBPA_H_matrix) ---
    rd_full = reshape(sarData, [], 1);    % (M*N) x 1
    rd      = rd_full(py);               % kk x 1

    % --- Sub-sampled propagation matrix Hp (kk x BA) ---
    Hp = H_bpa(py, :);                   % kk x (A*B)
    BA = A * B;

    % --- LIA core (myalg == 6 in Li & Chen SPL paper) ---
    di = 0.01;                           % initialization constant
    G  = di * (Hp' * Hp);                % (BA x BA)
    xd = di * (Hp' * rd);                % (BA x 1)

    % Iterative image updating based on matrix inversion lemma
    for j = 1:BA
        Gj    = G(:, j);                 % (BA x 1)
        denom = 1 + G(j, j);
        temp  = Gj / denom;              % (BA x 1)

        xd = xd - temp * xd(j);
        G  = G  - temp * G(j, :);
    end

    % ---- EXTRACT DIAGONAL AS COLUMN (dlarray-safe, no broadcasting) ----
    BA    = size(G, 1);
    diagG = G(1:BA+1:BA*BA);             % 1 x BA (row)
    diagG = reshape(diagG, [BA, 1]);     % BA x 1 (column)

    % Final scaling (element-wise)
    xd = xd ./ diagG;                    % stays BA x 1

    % Reshape to B x A and flip horizontally (as in original code)
    xdi = fliplr(reshape(xd, B, A));

    % --- Outputs ---
    trueImage_complx = xdi;

    % NO RMS-normalized magnitude
    img_mag = abs(trueImage_complx);
    % rms_val = sqrt(mean(img_mag(:).^2) + eps);
    % trueImage_abs = dlarray(img_mag ./ rms_val, 'SS');
    trueImage_abs = dlarray(img_mag, 'SS');

    % Spatial ranges (mm)
    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
end

%% %%%%%%%%%%%%%%%%%%%%% MFA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
function matchedFilter = refMF(params)
% REFMF: Generates the 2D Matched Filter coefficient.
    c  = physconst('lightspeed');
    x  = params.dx * (-(params.nFFTspace-1)/2 : (params.nFFTspace-1)/2) * 1e-3;
    y  = (params.dy * (-(params.nFFTspace-1)/2 : (params.nFFTspace-1)/2) * 1e-3).';
    z0_m = params.z0 * 1e-3;
    k  = 2 * pi * params.F0 / c;

    % Matched Filter: exp(-j * 2 * k * R)
    matchedFilter = exp(-1i * 2 * k * sqrt(bsxfun(@plus, x.^2, y.^2) + z0_m^2));
end

function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlMFA(sarData, params)
% DLMFA: 2D Matched Filter Algorithm for SAR image reconstruction.
    matchedFilter = refMF(params);
    if isa(sarData, 'dlarray') && ~isa(matchedFilter, 'dlarray')
        matchedFilter = dlarray(matchedFilter);
    end

    [yPointM, xPointM] = size(sarData);
    [yPointF, xPointF] = size(matchedFilter);

    % 1. Equalize Dimensions with Zero Padding
    % Pad sarData to match matchedFilter size (assuming nFFTspace >= M, N)
    if (xPointF > xPointM)
        pad_x_pre  = floor((xPointF - xPointM) / 2);
        pad_x_post = ceil((xPointF - xPointM) / 2);
        sarData    = cat(2, zeros(yPointM, pad_x_pre,  'like', sarData), ...
                            sarData, ...
                            zeros(yPointM, pad_x_post, 'like', sarData));
        xPointM = xPointF; % Update size
    end
    if (yPointF > yPointM)
        pad_y_pre  = floor((yPointF - yPointM) / 2);
        pad_y_post = ceil((yPointF - yPointM) / 2);
        sarData    = cat(1, zeros(pad_y_pre,  xPointM, 'like', sarData), ...
                            sarData, ...
                            zeros(pad_y_post, xPointM, 'like', sarData));
        yPointM = yPointF; % Update size
    end
    % Note: Padding matchedFilter to match sarData is omitted as sarData is padded up to nFFTspace.

    % 2. Run 2D FFT, Multiply, and 2D IFFT (Correlation)
    sarDataFFT        = fft(fft(sarData,        [], 2), [], 1);
    matchedFilterFFT  = fft(fft(matchedFilter,  [], 2), [], 1);

    trueImage_shifted = ifft(ifft(sarDataFFT .* matchedFilterFFT, [], 2), [], 1);

    % 3. Shift and Crop Image
    trueImage = fftshift(trueImage_shifted);

    [J, I] = size(trueImage);

    % Calculate crop indices based on bounding box
    xij = round(params.bbox(1:2) / params.dx - 0.5 + I/2);
    ykl = round(params.bbox(3:4) / params.dy - 0.5 + J/2);

    trueImage_cropped = trueImage(ykl(1):ykl(2), xij(1):xij(2));
    trueImage_cropped = fliplr(trueImage_cropped);
    trueImage_complx  = trueImage_cropped;

    % NO RMS-normalize magnitude output
    img_mag = abs(trueImage_cropped);
    %rms_val = sqrt(mean(img_mag(:).^2) + eps);
    %trueImage_abs = dlarray(img_mag ./ rms_val, 'SS');
    trueImage_abs = dlarray(img_mag, 'SS');

    % Spatial ranges for plotting
    xRangeT = params.bbox(1) + (0:size(trueImage_abs, 2) - 1) * params.dx;
    yRangeT = params.bbox(3) + (0:size(trueImage_abs, 1) - 1) * params.dy;
end

%% %%%%%%%%%%%%%%%%%%%%% BPA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    trueImage_complx  = trueImage_cropped; % Complex image output
    %trueImage_complx = xdi; % Complex image output

    % No RMS-normalize magnitude output
    img_mag = abs(trueImage_complx);
    % rms_val = sqrt(mean(img_mag(:).^2) + eps);
    % trueImage_abs = dlarray(img_mag ./ rms_val, 'SS'); % Magnitude output as dlarray
    trueImage_abs = dlarray(img_mag, 'SS');

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

    c0    = physconst('lightspeed');
    F0    = params.F0;     % Note: F0 is still needed here for the propagation constant
    z0_mm = params.z0;     % Target range in mm
    dx    = params.dx;     % Horizontal step in mm
    dy    = params.dy;     % Vertical step in mm
    bbox  = params.bbox;   % Bounding box in mm

    % --- 2. Convert to Meters (m) ---
    z0_m   = z0_mm * 1e-3;
    dxm    = dx * 1e-3;
    dym    = dy * 1e-3;
    bbox_m = bbox * 1e-3;

    % Constants
    % Propagation constant: j * 2 * k = j * 2 * (2*pi*F0/c0)
    k   = 2*pi*F0/c0;
    cst = 1i * 2 * k;
    %cst = sqrt(-1) * 2 * pi * F0 * 2 / c0;
    z2  = (z0_m)^2; % z^2 term for distance calculation

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
        ix = (i-1-iy) / N;  % Horizontal index (0 to M-1)

        % Sensor Coordinates (S_x, S_y) based on explicit indices
        sx_i = (ix + 0.5 - M/2) * dxm; % Sensor X coordinate (m)
        sy_i = (iy + 0.5 - N/2) * dym; % Sensor Y coordinate (m)

        % Iterate over all image pixels (columns, j)
        for j = 1:BA
            % Get pixel indices (jx, jy)
            jy = mod(j-1, B); jx = (j-1-jy) / B;

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

%% %%%%%%%%%%%%%%%%%%%%% RMA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlRMA(sarData, params)
% DLRMA: 2D Range Migration Algorithm for SAR image reconstruction using dlarray.
% This function is a dlarray-compatible version of the imaging_2DRMA core logic.

    % Extract parameters from struct
    nFFTspace = params.nFFTspace;
    z0_mm     = params.z0;
    dx        = params.dx;
    dy        = params.dy;
    bbox      = params.bbox;
    F0        = params.F0;

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
    kX  = linspace(-(wSx / 2), (wSx / 2), nFFTspace);
    wSy = 2 * pi / (dy * 1e-3);
    kY  = (linspace(-(wSy / 2), (wSy / 2), nFFTspace)).';

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
    K_mag_sq        = bsxfun(@plus, kX.^2, kY.^2);
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
        pad_x_pre  = floor((xPointF - xPointM) / 2);
        pad_x_post = ceil((xPointF - xPointM) / 2);
        % Use 'like' to maintain dlarray status if sarData is dlarray
        sarData = cat(2, zeros(yPointM, pad_x_pre, 'like', sarData), ...
                         sarData, ...
                         zeros(yPointM, pad_x_post, 'like', sarData));
    elseif (xPointM > xPointF)
        % Pad Phase Factor (shouldn't happen if nFFTspace is >= M, N)
        pad_x_pre  = floor((xPointM - xPointF) / 2);
        pad_x_post = ceil((xPointM - xPointF) / 2);
        phaseFactor = cat(2, zeros(yPointF, pad_x_pre, 'like', phaseFactor), ...
                             phaseFactor, ...
                             zeros(yPointF, pad_x_post, 'like', phaseFactor));
    end

    % Pad Y (Vertical)
    if (yPointF > yPointM)
        pad_y_pre  = floor((yPointF - yPointM) / 2);
        pad_y_post = ceil((yPointF - yPointM) / 2);
        sarData = cat(1, zeros(pad_y_pre,  size(sarData, 2), 'like', sarData), ...
                         sarData, ...
                         zeros(pad_y_post, size(sarData, 2), 'like', sarData));
    elseif (yPointM > yPointF)
        pad_y_pre  = floor((yPointM - yPointF) / 2);
        pad_y_post = ceil((yPointM - yPointF) / 2);
        phaseFactor = cat(1, zeros(pad_y_pre,  size(phaseFactor, 2), 'like', phaseFactor), ...
                             phaseFactor, ...
                             zeros(pad_y_post, size(phaseFactor, 2), 'like', phaseFactor));
    end

    % 4. 2D IFFT (Image Formation)
    %sarDataFFT = fft2(sarData, nFFTspace, nFFTspace);
    sarDataFFT = fft(fft(sarData, [], 2), [], 1);

    % Element-wise multiplication (Multiplication is gradient-aware)
    %trueImage_shifted = ifft2(sarDataFFT .* phaseFactor);
    trueImage = ifft(ifft(sarDataFFT .* phaseFactor, [], 2), [], 1);

    % Shift to center the image (like dlMFA)
    %trueImage = fftshift(trueImage_shifted);

    % 5. Crop and Normalize
    [J, I] = size(trueImage);

    % Calculate crop indices based on bounding box
    xij = round(bbox(1:2) / dx - 0.5 + I/2);
    ykl = round(bbox(3:4) / dy - 0.5 + J/2);

    trueImage_cropped = trueImage(ykl(1):ykl(2), xij(1):xij(2));
    trueImage_cropped = fliplr(trueImage_cropped);
    trueImage_complx  = trueImage_cropped;

    % No RMS-normalize magnitude output
    img_mag = abs(trueImage_cropped);
    %rms_val = sqrt(mean(img_mag(:).^2) + eps);
    %trueImage_abs = img_mag ./ rms_val;
    trueImage_abs = img_mag;

    % Set spatial ranges for plotting (consistent with dlMFA)
    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
end
