% -------------------------------------------------------------------------
% Differential Imaging (DI) Attack on Near-Field SAR Imaging (CSA Version)
% -------------------------------------------------------------------------
% Developed by:
%   Lhamo Dorje and Dr. Xiaohua (Edward) Li
%   ECE Department, SUNY Binghamton
%   November, 2025
%
% Description
%   This script implements a differential imaging (DI) attack
%   on Compressive Sensing Algorithm (CSA) / SBRIM-style reconstruction.
%
%   To make the optimization differentiable, we replace the iterative
%   SBRIM solver inside the loop with a linear Tikhonov inverse operator
%   W_csa built from the same forward model H_csa. The full CSA/SBRIM
%   imaging is still used for final visualization.
%
% Pipeline
%   1) Attack signal extraction:
%        - Align clean/attacked IQ via exhaustive delay + LS scaling.
%        - Extract residuals and build a large attack waveform pool X_aa.
%   2) Victim CSA imaging setup:
%        - Load raw SAR cube and true/target CSA images.
%        - Build CSA forward matrix H_csa and run numeric SBRIM (dlCSA).
%        - Precompute linear CSA imaging operator W_csa for gradients.
%   3) Attack construction and optimization:
%        - Select and frequency-align attack waveforms (X_a -> D).
%        - Optimize complex gains A via gradient descent:
%              A* = arg min_A ||I_csa(A) - I_target||_2^2 + λ_L2 ||A||_2^2
%   4) Final reconstruction and evaluation:
%        - Reconstruct attacked image via full CSA/SBRIM.
%        - Compute MSE, RMSE, NCC, SSIM, PSNR.
%        - Generate qualitative plots and spectral analysis.
%
% Required Files (on the MATLAB path)
%   - iqData_noAtk.mat  : iqData [Nsamp x nRX x nFrame] (clean IQ)
%   - iqData_Atk.mat    : iqData [Nsamp x nRX x nFrame] (attacked IQ)
%   - rawSAR.mat        : adcDataCube [Nsamp x M x N] (raw SAR cube)
%   - true_image_complex_CSA.mat         : trueImage_complx_csa [B x A]
%   - desired_attacked_complex_CSA.mat   : sar_camouflaged [B x A]
%
% Notes
%   - This script assumes near-field 2D SAR data with a single selected
%     range bin k0_range_bin.
%   - All images are RMS-normalized before computing MSE-based losses.
% -------------------------------------------------------------------------

clc; clear; close all;

%% ========================================================================
%  Step 1: Prepare & Extract Attack Signal Pool (X_aa)
% ========================================================================

sar_algo = 'CSA';   % Currently fixed to 'CSA' for this script

% --- Load IQ data and initialize parameters --------------------------------
try
    r0 = load("iqData_noAtk.mat").iqData;   % Nsamp x nRX x nFrame (clean IQ)
    r1 = load("iqData_Atk.mat").iqData;     % Nsamp x nRX x nFrame (attacked IQ)
catch ME
    error('Could not load IQ files. Place iqData_noAtk.mat and iqData_Atk.mat on the MATLAB path.\nMATLAB Error: %s', ...
          ME.message);
end

[Nsamp, nRX, nFrame] = size(r0);
p      = nRX * nFrame;
r0_vec = reshape(r0, Nsamp, p);  % Nsamp x p (vectorized across channels)
r1_vec = reshape(r1, Nsamp, p);

% --- Alignment parameters and storage -------------------------------------
tau_max_guess = 20;  % integer delay search window (samples)
tau_search    = max(-tau_max_guess, -(Nsamp-1)) : min(tau_max_guess, Nsamp-1);

r_v_est   = zeros(Nsamp, p, 'like', r0_vec);      % estimated victim: alpha * shifted r0
r_a_est   = zeros(Nsamp, p, 'like', r0_vec);      % attack residual: r1 - r_v_est
alpha_vec = complex(zeros(1, p, 'like', r0_vec)); % complex LS scaling factor
tau_samps = zeros(1, p);                          % selected integer delay

fprintf('Running exhaustive delay search + LS scaling on p=%d channels...\n', p);
tic;

% --- Main alignment loop: estimate per-channel alpha, tau, and residual ---
for col = 1:p
    r0_col = r0_vec(:, col);
    r1_col = r1_vec(:, col);

    % Skip if clean signal is near zero
    if norm(r0_col) < eps
        r_a_est(:, col) = r1_col;
        continue;
    end

    best_err        = Inf;
    best_alpha      = 0;
    best_tau        = 0;
    best_r0_shifted = zeros(Nsamp, 1, 'like', r0_col);

    % Exhaustive integer delay search + complex LS scaling
    for tau = tau_search
        % Zero-padded, non-circular shift
        if tau > 0
            r0_shifted = [zeros(tau, 1, 'like', r0_col); r0_col(1:end-tau)];
        elseif tau < 0
            t = -tau;
            r0_shifted = [r0_col(t+1:end); zeros(t, 1, 'like', r0_col)];
        else
            r0_shifted = r0_col;
        end

        % Complex LS scalar: alpha = (r0_shifted' r1_col) / ||r0_shifted||^2
        denom = (r0_shifted' * r0_shifted);
        alpha = (r0_shifted' * r1_col) / (denom + eps);

        % Residual error
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
    r_a_est(:, col)  = r1_col - r_v_est(:, col);  % extracted attack waveform
end

toc;
fprintf('Finished alignment. mean|alpha| = %.3e\n', mean(abs(alpha_vec)));

% --- Build attack pool X_aa (Nsamp x p^2) ---------------------------------
%   For each pair (i, j): X_aa(:, k) = r1_vec(:, j) - vhat(:, i)
vhat = r_v_est;         % vhat(:, i) = alpha_i * shifted r0(:, i)
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

% --- Column-wise RMS normalization for X_aa --------------------------------
colrms = sqrt(mean(abs(X_aa).^2, 1));
X_aa   = X_aa ./ (colrms + eps);

stat = @(X)[min(abs(X(:))), max(abs(X(:))), mean(abs(X(:)))];
fprintf('X_aa: size = %d x %d | min=%.3e  max=%.3e  mean=%.3e\n', ...
        size(X_aa, 1), size(X_aa, 2), stat(X_aa));

%% ========================================================================
%  Step 2: Load Victim SAR Data and Setup CSA Imaging Parameters
% ========================================================================

% File names for CSA true and target images
filename_clean    = sprintf('true_image_complex_%s.mat',  sar_algo);
filename_attacked = sprintf('desired_attacked_complex_%s.mat', sar_algo);

try
    % Raw SAR cube (victim measurements): Nsamp x M x N
    sarRawData = load('rawSAR.mat').adcDataCube;

    % Clean CSA image (reference)
    S_clean           = load(filename_clean);
    trueImage_complex = S_clean.trueImage_complx_csa;
    trueImage_complex = fliplr(trueImage_complex);

    % Desired attacked CSA image (target)
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

Echo = permute(sarRawData, [3, 2, 1]);   % [Nx, Nz, Nsamp]

X_v = reshape(sarRawData, Nsamp, M * N); % Nsamp x Np (victim raw data)
Np  = M * N;

% SAR system parameters
dx = 1; dy = 1;                         % aperture step (mm)
bbox = [-300 300 -300 300];            % imaging bounding box (mm)
c0 = physconst('lightspeed');
F0 = 77e9;                             % center frequency (Hz)
FS = 5000e3;                           % sampling rate (samples/s)
Ts = 1 / FS;                           % sampling period (s)
K0 = 70.295e12;                        % chirp slope (Hz/s)
tI = 4.5225e-10;                       % instrument delay (s)

num_sample = size(Echo, 3);
nFFTtime   = num_sample;
rawDataFFT = fft(Echo, nFFTtime, 3);   % FFT along fast-time

z0          = 185;  % target range (mm)
k0_range_bin = round(K0 * Ts * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);
sarData      = squeeze(rawDataFFT(:, :, k0_range_bin + 1)).';   % M x N

% Serpentine correction
for ii = 2:2:size(sarData, 1)
    sarData(ii, :) = fliplr(sarData(ii, :));
end

%% ========================================================================
%  Step 2.1: CSA Forward Model (H_csa) and Linear Surrogate (W_csa)
% ========================================================================

% Image grid size (CSA reconstruction)
A_pixels = 60;
B_pixels = 60;

% Parameter struct used throughout CSA and attack optimization
params = struct( ...
    'z0',           z0, ...
    'dx',           dx, ...
    'dy',           dy, ...
    'bbox',         bbox, ...
    'Nsamp',        Nsamp, ...
    'nFFTtime',     nFFTtime, ...
    'N',            N, ...
    'M',            M, ...
    'A',            A_pixels, ...
    'B',            B_pixels, ...
    'F0',           F0, ...
    'k0_range_bin', k0_range_bin, ...
    'sar_algo',     sar_algo);

% --- Build CSA forward matrix H_csa (M*N x A*B) ---------------------------
H_csa = dlCSA_H_matrix(params);   % dlarray

% CSA / SBRIM hyperparameters (for dlCSA numeric solver)
params.H_csa        = H_csa;
params.lambda0_csa  = 1e-4;
params.p_csa        = 1.0;
params.eta_csa      = 1e-5;
params.maxIter_csa  = 100;
params.epsilon0_csa = 1e-4;

% -------------------------------------------------------------------------
% Differentiable surrogate for optimization:
%   Backpropagating through the full SBRIM loop is expensive. We therefore
%   precompute a linear Tikhonov inverse operator
%
%       W_csa = (H^H H + λ_lin I)^{-1} H^H,
%
%   built from the same H_csa, and use it as a surrogate imaging operator
%   inside the gradient descent loop. The final attacked image is still
%   reconstructed with the full CSA/SBRIM algorithm (dlCSA).
% -------------------------------------------------------------------------

H_num     = double(extractdata(H_csa));             % (M*N x A*B)
lambda_lin = 1e-3;                                  % Tikhonov regularization
HtH       = H_num' * H_num;                         % (A*B x A*B)
W_csa_num = (HtH + lambda_lin * eye(size(HtH, 1))) \ (H_num');  % (A*B x M*N)

params.W_csa     = dlarray(W_csa_num);
params.lambda_lin = lambda_lin;

% --- CSA imaging of clean data (full SBRIM) for reference -----------------
[~, ~, trueImage_abs_csa, trueImage_complx_csa, alpha_hat] = dlCSA(sarData, params);

% RMS normalization for target images (ensures consistent MSE scale)
true_abs = extractdata(trueImage_abs_csa);
atk_abs  = abs(desired_attacked_complex);
rms_true = sqrt(mean(true_abs(:).^2) + eps);
rms_atk  = sqrt(mean(atk_abs(:).^2)  + eps);

params.trueImage       = dlarray(true_abs ./ rms_true, "SS");
params.desiredAtkImage = dlarray(atk_abs  ./ rms_atk,  "SS");

sarImage_clean   = extractdata(params.trueImage);
sarImage_true    = extractdata(params.trueImage);
sarImage_desired = extractdata(params.desiredAtkImage);

% All images use a shared color scale
top_val = max([sarImage_true(:); sarImage_desired(:)]);
clim    = [0, top_val];

figure('Name', 'Clean / True / Desired Images (CSA)', 'Color', 'w');
subplot(1, 3, 1);
plot_sar_bpa(sarImage_clean, bbox, dx, dy, ...
    sprintf('1. Clean Reconstructed Image (%s)', sar_algo));
caxis(clim); colorbar; colormap(jet);

subplot(1, 3, 2);
plot_sar_bpa(sarImage_true, bbox, dx, dy, ...
    sprintf('2. Loaded True Image (Reference, %s)', sar_algo));
caxis(clim); colorbar; colormap(jet);

subplot(1, 3, 3);
plot_sar_bpa(sarImage_desired, bbox, dx, dy, ...
    sprintf('3. Desired Attacked Image (Target, %s)', sar_algo));
caxis(clim); colorbar; colormap(jet);

%% ========================================================================
%  Step 3: Build Attack Raw Data and Optimize Complex Gains A
% ========================================================================

% --- 3.1: Select and frequency-align attack waveforms (X_a -> D) ----------
targetK    = 40000;                     % attack pool size to sample from
rng(0);                                 % reproducible sampling
sample_idx = randi(p * p, [1, targetK]);
X_a_pool   = X_aa(:, sample_idx);       % Nsamp x targetK

Np      = M * N;                        % number of aperture locations
sel_idx = randi(size(X_a_pool, 2), [1, Np]);
X_a     = X_a_pool(:, sel_idx);         % Nsamp x Np

% Per-column frequency shift toward victim range bin
Xspec   = fft(X_a, nFFTtime, 1);                 % Nsamp x Np
[~, b0] = max(abs(Xspec), [], 1);                % peak bin indices
f0      = (b0 - 1) * FS / nFFTtime;              % peak frequencies
f_tgt   = (params.k0_range_bin) * FS / nFFTtime; % target frequency
Delta   = f0 - f_tgt;                            % required shift

t = (0:Nsamp-1).' / FS;                          % time vector (s)
P = exp(-1j * 2*pi * (t * Delta));               % Nsamp x Np
D = P .* X_a;                                    % Nsamp x Np (aligned attacks)

% Scale D to match victim RMS on each aperture
colrms_fun = @(X) sqrt(mean(abs(X).^2, 1));
scale      = (colrms_fun(X_v) + eps) ./ (colrms_fun(D) + eps);
D          = D .* scale;

% --- 3.2: Gradient-based optimization of complex gains A ------------------
switch upper(sar_algo)
    case 'CSA'
        maxIter   = 50;      % increase if needed once stable
        lr_re     = 1e2;     % learning rate (real part of A)
        lr_im     = 1e2;     % learning rate (imag part of A)
        lambda_L2 = 1e-4;    % L2 regularization on |A|
    otherwise
        error('Invalid SAR algorithm selection.');
end

use_projection = true;   % optional hard magnitude cap on |A|
Amax           = 2;

% Initialization of A
A_re = dlarray(1e-3 * randn(Np, 1, 'double'));   % real part
A_im = dlarray(1e-3 * randn(Np, 1, 'double'));   % imaginary part

loss_hist  = zeros(maxIter, 1);
meanA_hist = zeros(maxIter, 1);
maxA_hist  = zeros(maxIter, 1);

% Live plots: loss, mean|A|, max|A|
figure('Name', 'Attack Optimization Progress (CSA)', 'Color', 'w');

subplot(3, 1, 1);
hLoss = semilogy(nan, nan, '-o');
grid on; xlabel('Iteration'); ylabel('Loss');
title('Loss vs Iteration');

subplot(3, 1, 2);
hMeanA = plot(nan, nan, '-o');
grid on; xlabel('Iteration'); ylabel('mean|A|');
title('Mean |A| vs Iteration');

subplot(3, 1, 3);
hMaxA = plot(nan, nan, '-o');
grid on; xlabel('Iteration'); ylabel('max|A|');
title('Max |A| vs Iteration');

fprintf('\nOptimizing complex gain A for all locations (Np=%d) with SAR algorithm: %s ...\n', ...
        Np, params.sar_algo);

for iter = 1:maxIter
    % Loss and gradients w.r.t. A_re, A_im
    [loss, gRe, gIm] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);

    % Convert loss to double for logging
    lossVal         = double(gather(extractdata(loss)));
    loss_hist(iter) = lossVal;

    % Gradient descent update
    A_re = A_re - lr_re * gRe;
    A_im = A_im - lr_im * gIm;

    % Optional magnitude projection: enforce |A| <= Amax
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

    % Record statistics on A
    A_now            = extractdata(A_re) + 1j * extractdata(A_im);
    meanA_hist(iter) = mean(abs(A_now));
    maxA_hist(iter)  = max(abs(A_now));

    % Live plot updates
    iters = 1:iter;
    set(hLoss,  'XData', iters, 'YData', loss_hist(1:iter));
    set(hMeanA, 'XData', iters, 'YData', meanA_hist(1:iter));
    set(hMaxA,  'XData', iters, 'YData', maxA_hist(1:iter));
    drawnow limitrate;

    % Console logging
    if mod(iter, 10) == 0 || iter == 1 || iter == maxIter
        fprintf('Iter %3d/%3d | Loss=%.6e | mean|A|=%.3e, max|A|=%.3e\n', ...
                iter, maxIter, lossVal, meanA_hist(iter), maxA_hist(iter));
    end
end
fprintf('-----------------------------\n');

%% ========================================================================
%  Step 3.3: Reconstruct Attacked Image with Optimal A (Full CSA/SBRIM)
% ========================================================================

A_opt = extractdata(A_re) + 1j * extractdata(A_im);   % Np x 1
Y_opt = X_v + D .* (A_opt.');                         % Nsamp x Np

% Reshape back to cube: Nsamp x M x N
Y_cube = reshape(Y_opt, Nsamp, M, N);

% Match CSA preprocessing pipeline (Echo -> FFT -> slice -> serpentine)
Echo_att       = permute(Y_cube, [3, 2, 1]);                % [N x M x Nsamp]
rawDataFFT_att = fft(Echo_att, nFFTtime, 3);                % FFT along fast-time
sarData_att    = squeeze(rawDataFFT_att(:, :, k0_range_bin + 1)).';   % [M x N]

for ii = 2:2:size(sarData_att, 1)
    sarData_att(ii, :) = fliplr(sarData_att(ii, :));
end

% Full CSA reconstruction (numeric SBRIM) of attacked data
[~, ~, atkImage_abs_csa, ~, ~] = dlCSA(sarData_att, params);
I_att = extractdata(atkImage_abs_csa);

figure('Name', 'Attacked Image (A_{opt}, CSA)', 'Color', 'w');
plot_sar_bpa(I_att, params.bbox, params.dx, params.dy, ...
    sprintf('Attacked Image (Reconstructed using A_{opt}, %s)', params.sar_algo));
caxis(clim); colorbar; colormap(jet);

%% ========================================================================
%  Step 4: Quantitative Attack Evaluation (CSA)
% ========================================================================

I_ref = extractdata(params.desiredAtkImage);  % target (desired attacked image)

fprintf('\n--- Spoofing Attack Performance (CSA) ---\n');

% 1) Mean Squared Error (MSE)
mse_val = mean((I_att(:) - I_ref(:)).^2);
fprintf('Mean Squared Error (MSE): %.6e\n', mse_val);

% 2) Root Mean Squared Error (RMSE)
rmse_val = sqrt(mse_val);
fprintf('Root Mean Squared Error (RMSE): %.6e\n', rmse_val);

% 3) Normalized Cross-Correlation (NCC)
I_ref_vec = I_ref(:);
I_att_vec = I_att(:);
numerator   = sum(I_ref_vec .* I_att_vec);
denominator = sqrt(sum(I_ref_vec.^2) * sum(I_att_vec.^2));
ncc_val     = numerator / (denominator + eps);
fprintf('Normalized Cross-Correlation (NCC): %.4f\n', ncc_val);

% 4) Structural Similarity Index (SSIM)
if license('test', 'Image_Toolbox')
    ssim_val = ssim(I_att, I_ref, 'DynamicRange', max(I_ref(:)));
    fprintf('Structural Similarity Index (SSIM): %.4f\n', ssim_val);
else
    fprintf('Structural Similarity Index (SSIM): Skipped (Image Processing Toolbox required).\n');
    ssim_val = NaN;
end

% 5) Peak Signal-to-Noise Ratio (PSNR)
if license('test', 'Image_Toolbox')
    max_val  = max(I_ref(:));
    psnr_val = psnr(I_att, I_ref, max_val);
    fprintf('Peak Signal-to-Noise Ratio (PSNR): %.2f dB\n', psnr_val);
else
    fprintf('Peak Signal-to-Noise Ratio (PSNR): Skipped (Image Processing Toolbox required).\n');
end

%% ========================================================================
%  Step 5: Visual Comparison and Spectral Analysis (CSA)
% ========================================================================

sarImage_clean = extractdata(params.trueImage);  % normalized clean CSA image
algo_name      = upper(params.sar_algo);

% Axes for plotting in mm
xv = params.bbox(1) + (0:size(I_ref, 2)-1) * params.dx;
yv = params.bbox(3) + (0:size(I_ref, 1)-1) * params.dy;

% (a) Clean image
fig_clean = figure('Color', 'w');
imagesc(xv, yv, sarImage_clean);
set(gca, 'YDir', 'normal'); axis image;
colormap(gca, 'jet'); caxis(clim);
xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
colorbar; title('Clean (True Image, CSA)');

% (b) Target image
fig_ref = figure('Color', 'w');
imagesc(xv, yv, I_ref);
set(gca, 'YDir', 'normal'); axis image;
colormap(gca, 'jet'); caxis(clim);
xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
colorbar; title('Target (Desired Attacked, CSA)');

% (c) Attacked image
fig_att = figure('Color', 'w');
imagesc(xv, yv, I_att);
set(gca, 'YDir', 'normal'); axis image;
colormap(gca, 'jet'); caxis(clim);
xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
colorbar; title('Reconstructed (After Attack, CSA)');

% (d) Error map |I_att - I_ref|
err_map = abs(I_att - I_ref);
fig_err = figure('Color', 'w');
imagesc(xv, yv, err_map);
set(gca, 'YDir', 'normal'); axis image;
colormap(gca, 'jet');
xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
colorbar; title('|I_{att} - I_{ref}| (Error Map, CSA)');

% Side-by-side comparison panel
figure('Name', 'Attack Evaluation: Image Comparison (CSA)', ...
       'Units', 'normalized', 'Position', [0.05 0.1 0.9 0.45]);

subplot(1, 4, 1);
imagesc(sarImage_clean); axis image off;
title('Clean (True Image)'); colormap jet; colorbar;

subplot(1, 4, 2);
imagesc(I_ref); axis image off;
title('Target (Desired Attacked)'); colormap jet; colorbar;

subplot(1, 4, 3);
imagesc(I_att); axis image off;
title('Reconstructed (After Attack)'); colormap jet; colorbar;

subplot(1, 4, 4);
imagesc(abs(I_att - I_ref)); axis image off;
title('|I_{att} - I_{ref}|'); colormap hot; colorbar;
sgtitle(sprintf('CSA Spoofing Attack (NCC=%.3f, SSIM=%.3f)', ncc_val, ssim_val));

% Pixel-wise correlation scatter plot
figure('Name', 'Pixel Correlation (CSA)', ...
       'Units', 'normalized', 'Position', [0.3 0.3 0.4 0.4]);
scatter(I_ref(:), I_att(:), 5, '.'); hold on;
maxVal = max([max(I_ref(:)), max(I_att(:))]);
pad    = 0.05 * maxVal;
plot([0, maxVal+pad], [0, maxVal+pad], 'r--', 'LineWidth', 1.2);
xlabel('Target Intensity I_{ref}');
ylabel('Attacked Intensity I_{att}');
axis equal; grid on;
xlim([0, maxVal+pad]);
ylim([0, maxVal+pad]);
title(sprintf('Pixel Correlation (CSA) (NCC=%.3f, SSIM=%.3f)', ncc_val, ssim_val));

% Error histogram
figure('Name', 'Error Histogram (CSA)', ...
       'Units', 'normalized', 'Position', [0.3 0.3 0.4 0.4]);
histogram(I_att(:) - I_ref(:), 100);
xlabel('Pixel Error (I_{att} - I_{ref})');
ylabel('Count');
title('Error Distribution (CSA)');
grid on;

% Compact error histogram with fixed x-limits
err = I_att(:) - I_ref(:);
figure('Name', 'Pixel Error Histogram (Limited Range, CSA)', ...
       'Color', 'w', ...
       'Units', 'normalized', ...
       'Position', [0.25 0.35 0.50 0.25]);
histogram(err, 80);
grid on;
xlim([-1.5 1.5]);
xlabel('Pixel error  (I_{att} - I_{ref})', 'Interpreter', 'tex');
ylabel('Number of pixels', 'Interpreter', 'tex');
set(gca, 'FontSize', 12);

% Log-spectrum comparison
figure('Name', 'Log-Spectrum Comparison (CSA)', ...
       'Units', 'normalized', 'Position', [0.05 0.1 0.9 0.4]);
subplot(1, 3, 1);
imagesc(log10(abs(fftshift(fft2(I_ref))) + 1e-12)); axis image off;
title('Target Spectrum'); colormap jet; colorbar;

subplot(1, 3, 2);
imagesc(log10(abs(fftshift(fft2(I_att))) + 1e-12)); axis image off;
title('Attacked Spectrum'); colormap jet; colorbar;

subplot(1, 3, 3);
imagesc(log10(abs(fftshift(fft2(I_att))) + 1e-12) - ...
        log10(abs(fftshift(fft2(I_ref))) + 1e-12));
axis image off; colormap jet; colorbar;
title('Spectral Difference (dB)');
sgtitle('Frequency-Domain Similarity (CSA)');

%% ========================================================================
%  Helper Functions (CSA Imaging, Forward Model, and Plotting)
% ========================================================================

function [loss, gradRe, gradIm] = loss_and_grad(X_v, D, A_re, A_im, params, lambda_L2)
% LOSS_AND_GRAD
%   Computes the MSE image loss plus L2 regularization, and returns
%   gradients with respect to the real and imaginary parts of A.
%
%   Inputs
%     X_v       : Nsamp x Np (clean victim raw data)
%     D         : Nsamp x Np (frequency-aligned attack waveforms)
%     A_re      : Np x 1 (real part of complex gains A)
%     A_im      : Np x 1 (imag part of complex gains A)
%     params    : struct with CSA and SAR parameters (incl. W_csa)
%     lambda_L2 : L2 regularization weight on |A|^2
%
%   Outputs
%     loss   : scalar dlarray loss
%     gradRe : gradient d(loss)/d(A_re)
%     gradIm : gradient d(loss)/d(A_im)

    if ~isa(X_v, 'dlarray'), X_v = dlarray(X_v); end
    if ~isa(D,   'dlarray'), D   = dlarray(D);   end
    if nargin < 6 || isempty(lambda_L2), lambda_L2 = 0; end

    % Complex gains and attacked raw data
    A = A_re + 1j * A_im;            % Np x 1
    Y = X_v + D .* A.';              % Nsamp x Np

    % Reshape to cube: Nsamp x M x N
    Y_cube = reshape(Y, params.Nsamp, params.M, params.N);

    % CSA preprocessing: EchoY -> FFT -> single range bin -> serpentine
    EchoY    = permute(Y_cube, [3, 2, 1]);                         % [N x M x Nsamp]
    rawDataFFT = fft(EchoY, params.nFFTtime, 3);                   % FFT along fast-time
    sarData = squeeze(rawDataFFT(:, :, params.k0_range_bin+1)).';  % [M x N]

    for ii = 2:2:size(sarData, 1)
        sarData(ii, :) = fliplr(sarData(ii, :));
    end

    % Vectorize measurements
    ys = reshape(sarData, [], 1);        % (M*N x 1), dlarray

    % Linear CSA imaging surrogate: α_hat = W_csa * ys
    alpha_hat_vec    = params.W_csa * ys;      % (A*B x 1)
    B                = params.B;
    A_sz             = params.A;
    atkImage_complex = reshape(alpha_hat_vec, B, A_sz);   % B x A

    atk_mag  = abs(atkImage_complex);
    rms_atk  = sqrt(mean(atk_mag(:).^2) + eps);
    atkImage = atk_mag ./ rms_atk;                     % normalized magnitude image

    % Image-domain loss
    loss_im = mean((atkImage - params.desiredAtkImage).^2, 'all');

    % L2 regularization on A
    reg  = lambda_L2 * mean(abs(A).^2, 'all');
    loss = loss_im + reg;

    % Gradients w.r.t. A_re, A_im
    [gradRe, gradIm] = dlgradient(loss, A_re, A_im);
end

function [xRangeT, yRangeT, trueImage_abs, trueImage_complx, alpha_hat_dl] = dlCSA(sarData, params)
% DLCSA
%   CSA imaging using a numeric SBRIM solver, with dlarray outputs.
%
%   Inputs
%     sarData : M x N complex slice at the chosen range bin
%     params  : struct containing H_csa and CSA hyperparameters
%
%   Outputs
%     xRangeT, yRangeT   : spatial axes (mm)
%     trueImage_abs      : RMS-normalized magnitude image (dlarray, 'SS')
%     trueImage_complx   : complex image (B x A, dlarray)
%     alpha_hat_dl       : vectorized complex image coefficients (dlarray)

    % Convert sarData to numeric
    if isa(sarData, 'dlarray')
        sarData_num = double(extractdata(sarData));
    else
        sarData_num = double(sarData);
    end

    % Convert H_csa to numeric
    if isa(params.H_csa, 'dlarray')
        H_num = double(extractdata(params.H_csa));
    else
        H_num = double(params.H_csa);
    end

    % Vectorize measurements
    ys_num = sarData_num(:);     % (M*N x 1)

    % Numeric SBRIM solver
    alpha_hat_num = CSA_SBRIM_numeric(ys_num, H_num, ...
                                      params.lambda0_csa, ...
                                      params.p_csa, ...
                                      params.eta_csa, ...
                                      params.maxIter_csa, ...
                                      params.epsilon0_csa);

    % Reshape to 2D image
    B = params.B;
    A = params.A;
    alpha_img_num = reshape(alpha_hat_num, B, A);   % B x A

    % Wrap as dlarray
    trueImage_complx = dlarray(alpha_img_num);
    img_mag          = abs(alpha_img_num);
    rms_val          = sqrt(mean(img_mag(:).^2) + eps);
    trueImage_abs    = dlarray(img_mag ./ rms_val, 'SS');

    alpha_hat_dl = dlarray(alpha_hat_num);

    % Spatial ranges (mm)
    xRangeT = params.bbox(1) + (0:A-1) * params.dx;
    yRangeT = params.bbox(3) + (0:B-1) * params.dy;
end

function alpha_hat = CSA_SBRIM_numeric(ys, H, lambda0, p, eta, maxIter, epsilon0)
% CSA_SBRIM_NUMERIC
%   Numeric SBRIM solver (no dlarray) for CSA imaging:
%
%     min_α ||y - Hα||_2^2 + λ0 β ∑_i (|α_i|^2 + η)^(p/2)
%
%   The method updates α and the noise variance β iteratively until the
%   relative change r = ||α^{(n)} - α^{(n-1)}|| / ||α^{(n)}|| drops below
%   epsilon0 or maxIter is reached.

    ys = double(ys);
    H  = double(H);

    [M_meas, N_pixels] = size(H);

    % Precompute H'H and H'y
    temp1 = H' * H;     % N x N
    HH_ys = H' * ys;    % N x 1

    % Initialize with matched filter solution
    alpha_hat_prev = HH_ys;
    alpha_hat      = alpha_hat_prev;
    r      = Inf;
    n      = 0;
    beta_n = 1;

    fprintf('Starting SBRIM (numeric) with N_pixels=%d, p=%.2f...\n', N_pixels, p);

    while (r >= epsilon0) && (n < maxIter)
        n = n + 1;
        alpha_hat_prev = alpha_hat;

        % λ_i = (p/2) (|α_i|^2 + η)^{p/2 - 1}
        alpha_sq_plus_eta = abs(alpha_hat_prev).^2 + eta;
        lambda_diag       = (p / 2) * (alpha_sq_plus_eta).^(p/2 - 1);   % N x 1

        Lambda_n = diag(lambda_diag);

        % Regularized normal matrix
        A_mat = temp1 + lambda0 * beta_n * Lambda_n;

        % Solve for α
        alpha_hat = A_mat \ HH_ys;

        % Update β based on residual
        residual = ys - H * alpha_hat;
        beta_n   = sum(abs(residual).^2) / M_meas;

        % Convergence ratio
        norm_alpha_n = norm(alpha_hat);
        if norm_alpha_n < eps
            r = 0;
        else
            r = norm(alpha_hat - alpha_hat_prev) / norm_alpha_n;
        end

        if mod(n, 10) == 0 || n == 1
            fprintf('Iter %d: r=%.4e, beta=%.4e\n', n, r, beta_n);
        end
    end

    if n == maxIter
        fprintf('Warning: SBRIM reached maxIter=%d (r=%.4e)\n', maxIter, r);
    else
        fprintf('SBRIM converged in %d iters (r=%.4e)\n', n, r);
    end
end

function H = dlCSA_H_matrix(params)
% DLCSA_H_MATRIX
%   Builds the propagation matrix H (Measurements x Pixels) for the CSA
%   / BPA-style forward model:
%
%     y(m) = ∑_n H(m, n) α(n),
%
%   where H encodes near-field propagation from image pixels (B x A) to
%   aperture positions (M x N).
%
%   Inputs (in params)
%     M, N : aperture dimensions
%     A, B : image dimensions
%     F0   : center frequency
%     z0   : target range (mm)
%     dx, dy : aperture step (mm)
%     bbox : [xmin xmax ymin ymax] in mm
%
%   Output
%     H : (M*N x A*B) propagation matrix, as a dlarray.

    M = params.M;       % aperture horizontal points
    N = params.N;       % aperture vertical points
    A = params.A;       % image horizontal pixels
    B = params.B;       % image vertical pixels

    c0    = physconst('lightspeed');
    F0    = params.F0;      % center frequency
    z0_mm = params.z0;      % target range in mm
    dx    = params.dx;      % aperture step in mm
    dy    = params.dy;
    bbox  = params.bbox;    % [xmin xmax ymin ymax] in mm

    % Convert to meters
    z0_m   = z0_mm * 1e-3;
    dxm    = dx   * 1e-3;
    dym    = dy   * 1e-3;
    bbox_m = bbox * 1e-3;

    % Propagation constant: 2k = 2 * (2π F0 / c0)
    k   = 2 * pi * F0 / c0;
    cst = 1i * 2 * k;
    z2  = z0_m^2;

    % Image pixel coordinates (P_x, P_y)
    wh1 = linspace(bbox_m(1), bbox_m(2), A);  % horizontal pixels (m)
    wh2 = linspace(bbox_m(3), bbox_m(4), B);  % vertical pixels (m)

    % Sensor/aperture coordinates (S_x, S_y)
    [ix_vec, iy_vec] = meshgrid(0:M-1, 0:N-1);
    sx = (ix_vec(:) + 0.5 - M/2) * dxm;       % M*N x 1
    sy = (iy_vec(:) + 0.5 - N/2) * dym;       % M*N x 1

    NM = M * N;   % number of measurements
    BA = A * B;   % number of pixels

    H_val = complex(zeros(NM, BA));

    fprintf('    Building H matrix (%d x %d)...', NM, BA);
    tic;

    for i = 1:NM
        iy = mod(i-1, N);           % 0..N-1
        ix = (i-1-iy) / N;          % 0..M-1

        sx_i = (ix + 0.5 - M/2) * dxm;
        sy_i = (iy + 0.5 - N/2) * dym;

        for j = 1:BA
            jy = mod(j-1, B);
            jx = (j-1-jy) / B;

            px = wh1(jx+1);
            py = wh2(jy+1);

            dist2 = (sx_i - px)^2 + (sy_i - py)^2 + z2;
            H_val(i, j) = exp(cst * sqrt(dist2));
        end
    end

    fprintf(' %.3f sec\n', toc);
    H = dlarray(H_val);
end


function plot_sar_bpa(I_plot, bbox, dx, dy, plotTitle)
% PLOT_SAR_BPA
%   Helper visualization: plots a 2D SAR image using mesh + view(2),
%   with axes in millimeters defined by bbox.

    [B, A] = size(I_plot);

    xv = linspace(bbox(1), bbox(2), A);  % horizontal axis (mm)
    yv = linspace(bbox(3), bbox(4), B);  % vertical axis (mm)

    mesh(xv, yv, I_plot, 'FaceColor', 'interp', 'LineStyle', 'none');
    view(2);
    colormap('jet');
    axis equal tight;

    xlabel('Horizontal (mm)');
    ylabel('Vertical (mm)');
    title(plotTitle);

    xlim([bbox(1) bbox(2)]);
    ylim([bbox(3) bbox(4)]);
end