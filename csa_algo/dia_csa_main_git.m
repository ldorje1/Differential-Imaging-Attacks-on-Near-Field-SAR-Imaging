clc; clear; close all;

%% ========================================================================
%  CSA Differential Imaging Attack (DIA) on real SAR measurements
%
%  Loads a victim raw data cube, gates a selected range bin, and forms a
%  clean CSA/SBRIM reconstruction. A structured target is created by
%  shuffling the gated measurements and re-imaging. An attack waveform pool
%  is sampled, frequency-aligned to the victim range bin, and RMS-matched to
%  the victim measurements to build an injection dictionary D. Complex
%  per-aperture gains A are then optimized via backpropagation through a
%  differentiable CSA linear surrogate to minimize MSE to the target, with
%  optional |A| projection and L2 regularization. The final attacked image
%  is reconstructed using the full CSA/SBRIM solver and evaluated (MSE, NCC,
%  SSIM, PSNR) with visualization.
% ========================================================================

sar_algo = 'CSA';                       % fixed for this script
dataDir  = fullfile(pwd, 'data');

%% ========================================================================
%  Load victim SAR data and CSA imaging parameters
% ========================================================================

sarRawData = load(fullfile(dataDir, 'rawSAR.mat')).adcDataCube;
[Nsamp, M, N] = size(sarRawData);

Echo = permute(sarRawData, [3, 2, 1]);               % [Nx, Nz, Nsamp]
X_v  = reshape(sarRawData, Nsamp, M*N);              % Nsamp x Np
Np   = M * N;

% SAR system parameters
dx   = 1;  dy   = 1;                                 % (mm)
bbox = [-300 300 -300 300];                          % (mm)
c0   = physconst('lightspeed');
F0   = 77e9;                                         % (Hz)
FS   = 5000e3;                                       % (samples/s)
Ts   = 1 / FS;
K0   = 70.295e12;                                    % (Hz/s)
tI   = 4.5225e-10;                                   % (s)

% FFT along fast-time
nFFTtime   = size(Echo, 3);
rawDataFFT = fft(Echo, nFFTtime, 3);

% Range-bin selection
z0           = 185;                                  % (mm)
k0_range_bin = round(K0 * Ts * (2*z0*1e-3/c0 + tI) * nFFTtime);

%% ========================================================================
%  CSA forward model and linear surrogate
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

% CSA forward matrix H_csa (M*N x A*B)
H_csa = dlCSA_H_matrix(params);                      % dlarray

% CSA / SBRIM hyperparameters (for dlCSA numeric solver)
params.H_csa        = H_csa;
params.lambda0_csa  = 1e-4;
params.p_csa        = 1.0;
params.eta_csa      = 1e-5;
params.maxIter_csa  = 100;
params.epsilon0_csa = 1e-4;

% Differentiable linear surrogate:
%   W_csa = (H^H H + λ_lin I)^{-1} H^H
H_num      = double(extractdata(H_csa));             % (M*N x A*B)
lambda_lin = 1e-3;
HtH        = H_num' * H_num;                        % (A*B x A*B)
W_csa_num  = (HtH + lambda_lin * eye(size(HtH,1))) \ (H_num');  % (A*B x M*N)

params.W_csa      = dlarray(W_csa_num);
params.lambda_lin = lambda_lin;

%% ========================================================================
%  Clean and shuffled target reconstructions (CSA)
% ========================================================================

% Gate the chosen bin and transpose to M x N
sarData = squeeze(rawDataFFT(:, :, k0_range_bin + 1)).';          % M x N

% Serpentine correction (flip every other scan row)
for ii = 2:2:size(sarData, 1)
    sarData(ii, :) = fliplr(sarData(ii, :));
end

% Clean image (full SBRIM)
[~, ~, clean_img, ~, ~] = dlCSA(sarData, params);
clean_img = extractdata(clean_img);

% Target image (structured random): shuffle gated+serpentine sarData
rng(42);
sarData_vec      = sarData(:);
sarData_shuffled = reshape(sarData_vec(randperm(numel(sarData_vec))), size(sarData));

[~, ~, target_img, ~] = dlCSA(sarData_shuffled, params);
target_img = extractdata(target_img);

% Global normalization
global_scale = max(abs(clean_img(:))) + 1e-12;

clean_img  = clean_img  / global_scale;
target_img = target_img / global_scale;

params.global_scale = global_scale;
params.clean_img    = dlarray(clean_img,  "SS");
params.target_img   = dlarray(target_img, "SS");

% Quick visualization: clean vs shuffled target
figure();

subplot(1,2,1);
imagesc(abs(clean_img)); axis image off; colormap gray; colorbar;
title(sprintf('Clean Image (%s)', upper(sar_algo)));

subplot(1,2,2);
imagesc(abs(target_img)); axis image off; colormap gray; colorbar;
title(sprintf('Target Image (%s) - Shuffled', upper(sar_algo)));

%% ========================================================================
%  Attack waveform construction and gain optimization
% ========================================================================

% Load attack signal pool (X_aa)
temp_x_aa = load(fullfile(dataDir, "X_aa.mat"));     % loads a .mat file that contains X_aa
X_aa      = temp_x_aa.X_aa;                          % Nsamp x (pool_size) complex waveforms

% Select and frequency-align attack waveforms (X_a -> D)
targetK    = 40000;                                  % attack pool size to sample from
rng(0);                                              % reproducible sampling
sample_idx = randi(size(X_aa,2), [1, targetK]);
X_a_pool   = X_aa(:, sample_idx);                    % Nsamp x targetK

Np      = M * N;                                     % number of aperture locations
sel_idx = randi(size(X_a_pool, 2), [1, Np]);
X_a     = X_a_pool(:, sel_idx);                      % Nsamp x Np

% Per-column frequency shift toward victim range bin
Xspec   = fft(X_a, nFFTtime, 1);                      % Nsamp x Np
[~, b0] = max(abs(Xspec), [], 1);                     % peak bin indices
f0      = (b0 - 1) * FS / nFFTtime;                   % peak frequencies
f_tgt   = (params.k0_range_bin) * FS / nFFTtime;      % target frequency
Delta   = f0 - f_tgt;                                 % required shift

t = (0:Nsamp-1).' / FS;                               % time vector (s)
P = exp(-1j * 2*pi * (t * Delta));                    % Nsamp x Np
D = P .* X_a;                                         % Nsamp x Np (aligned attacks)

% RMS matching to victim measurements (per aperture)
colrms_fun = @(X) sqrt(mean(abs(X).^2, 1));
scale      = (colrms_fun(X_v) + eps) ./ (colrms_fun(D) + eps);
D          = D .* scale;

%% ---------------------------------------------------------------
% DIA Optimization
% ---------------------------------------------------------------
maxIter   = 300;                                      % increase if needed once stable
lr_re     = 1e4;                                      % learning rate (real part of A)
lr_im     = 1e4;                                      % learning rate (imag part of A)
lambda_L2 = 1e-5;                                     % L2 regularization on |A|

use_projection = true;                                % optional hard magnitude cap on |A|
Amax           = 2;

% Initialization of A
A_re = dlarray(1e-3 * randn(Np, 1, 'double'));        % real part
A_im = dlarray(1e-3 * randn(Np, 1, 'double'));        % imaginary part

fprintf(['\nOptimizing complex gain A for all locations (Np=%d) with SAR algorithm: %s ' ...
         '| lr_re=%.3g, lr_im=%.3g | lambda_L2=%.3g | proj=%d\n'], ...
        Np, params.sar_algo, lr_re, lr_im, lambda_L2, use_projection);

for iter = 1:maxIter

    % IMPORTANT: loss_and_grad must return atkImage as 4th output
    [loss, gRe, gIm, atkImage] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);

    lossVal = double(gather(extractdata(loss)));

    % Gradient update
    A_re = A_re - lr_re * gRe;
    A_im = A_im - lr_im * gIm;

    % Projection
    if use_projection
        A_num = extractdata(A_re) + 1j * extractdata(A_im);
        mags  = abs(A_num);
        over  = mags > Amax;
        if any(over)
            scale_proj        = ones(size(mags));
            scale_proj(over)  = Amax ./ mags(over);
            A_num             = A_num .* scale_proj;
            A_re              = dlarray(real(A_num));
            A_im              = dlarray(imag(A_num));
        end
    end

    if mod(iter, 50) == 0 || iter == 1 || iter == maxIter

        % A_log: current A in numeric form (post-projection if enabled)
        A_log = extractdata(A_re) + 1j*extractdata(A_im);

        % Summary stats of A
        meanA = mean(abs(A_log), 'all');
        maxA  = max(abs(A_log),  [], 'all');

        % MSE between attacked image and target/clean (atkImage is a dlarray)
        mse_AT = double(gather(extractdata(mean((atkImage - params.target_img).^2, 'all'))));
        mse_CA = double(gather(extractdata(mean((atkImage - params.clean_img ).^2, 'all'))));

        % Attack-to-victim power ratio Pa/Pr in the signal domain (Frobenius norm squared)
        delta_now = D .* (A_log.');                                % Nsamp x Np injected signal
        PaPr      = (norm(delta_now,'fro') / (norm(X_v,'fro') + 1e-12))^2;

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

%% ========================================================================
%  Attacked reconstruction and evaluation
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
[~, ~, atkImage_abs, ~, ~] = dlCSA(sarData_att, params);
adv_img = extractdata(atkImage_abs);

%% ---------------------------------------------------------------
% Final metric evaluation (image domain)
% ---------------------------------------------------------------
adv_img = adv_img / params.global_scale;   % only if you have it

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

if exist('ssim','file') == 2
    ssim_val = ssim(adv_img, target_img, 'DynamicRange', data_range);
else
    ssim_val = NaN;
end

if exist('psnr','file') == 2
    psnr_val = psnr(adv_img, target_img, data_range);
else
    psnr_val = 10*log10((data_range^2) / (mse_val + 1e-12));
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

subplot(2,2,1);
imagesc(clean_img); axis image off; colormap gray; colorbar;
title(sprintf('Clean %s output image', upper(params.sar_algo)));

subplot(2,2,2);
imagesc(target_img); axis image off; colormap gray; colorbar;
title('Target image (desired attacked)');

subplot(2,2,3);
imagesc(adv_img); axis image off; colormap gray; colorbar;
title('Adversarial image (attacked)');

subplot(2,2,4);
imagesc(adv_img - clean_img); axis image off; colormap gray; colorbar;
title('Diff (A-C)');

%% ========================================================================
%  Helper Functions (CSA Imaging, Forward Model, and Plotting)
% ========================================================================

function [loss, gradRe, gradIm, atkImage] = loss_and_grad(X_v, D, A_re, A_im, params, lambda_L2)
% LOSS_AND_GRAD
%   Computes MSE in the image domain (attacked vs target) plus L2
%   regularization on |A|, and returns gradients w.r.t. A_re and A_im.

    if ~isa(X_v, 'dlarray'), X_v = dlarray(X_v); end
    if ~isa(D,   'dlarray'), D   = dlarray(D);   end
    if nargin < 6 || isempty(lambda_L2), lambda_L2 = 0; end

    % Complex gains and attacked raw data
    A = A_re + 1j * A_im;            % Np x 1
    Y = X_v + D .* A.';              % Nsamp x Np

    % Reshape to cube: Nsamp x M x N
    Y_cube = reshape(Y, params.Nsamp, params.M, params.N);

    % CSA preprocessing: EchoY -> FFT -> single range bin -> serpentine
    EchoY      = permute(Y_cube, [3, 2, 1]);                         % [N x M x Nsamp]
    rawDataFFT = fft(EchoY, params.nFFTtime, 3);                     % FFT along fast-time
    sarData    = squeeze(rawDataFFT(:, :, params.k0_range_bin+1)).'; % [M x N]

    for ii = 2:2:size(sarData, 1)
        sarData(ii, :) = fliplr(sarData(ii, :));
    end

    % Vectorize measurements
    ys = reshape(sarData, [], 1);                                    % (M*N x 1), dlarray

    % Linear CSA imaging surrogate: α_hat = W_csa * ys
    alpha_hat_vec    = params.W_csa * ys;                            % (A*B x 1)
    B                = params.B;
    A_sz             = params.A;
    atkImage_complex = reshape(alpha_hat_vec, B, A_sz);              % B x A

    atkImage = abs(atkImage_complex);
    atkImage = atkImage / params.global_scale;

    % Image-domain loss
    loss_im = mean((atkImage - params.target_img).^2, 'all');

    % L2 regularization on A
    reg  = lambda_L2 * mean(abs(A).^2, 'all');
    loss = loss_im + reg;

    % Gradients w.r.t. A_re, A_im
    [gradRe, gradIm] = dlgradient(loss, A_re, A_im);
end

function [xRangeT, yRangeT, trueImage_abs, trueImage_complx, alpha_hat_dl] = dlCSA(sarData, params)
% DLCSA
%   CSA imaging using a numeric SBRIM solver, with dlarray outputs.

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
    ys_num = sarData_num(:);                                      % (M*N x 1)

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
    alpha_img_num = reshape(alpha_hat_num, B, A);                 % B x A

    % Wrap as dlarray
    trueImage_complx = dlarray(alpha_img_num);
    img_mag          = abs(alpha_img_num);
    trueImage_abs    = dlarray(img_mag, 'SS');

    alpha_hat_dl = dlarray(alpha_hat_num);

    % Spatial ranges (mm)
    xRangeT = params.bbox(1) + (0:A-1) * params.dx;
    yRangeT = params.bbox(3) + (0:B-1) * params.dy;
end

function alpha_hat = CSA_SBRIM_numeric(ys, H, lambda0, p, eta, maxIter, epsilon0)
% CSA_SBRIM_NUMERIC
%   Numeric SBRIM solver for CSA imaging.

    ys = double(ys);
    H  = double(H);

    [M_meas, N_pixels] = size(H);

    % Precompute H'H and H'y
    temp1 = H' * H;                                               % N x N
    HH_ys = H' * ys;                                              % N x 1

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

        % λ_i = (p/2) (|α_i|^2 + η)^(p/2 - 1)
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
%   Builds the propagation matrix H (M*N x A*B) mapping image pixels to
%   aperture samples under a near-field phase model.

    M = params.M;                                            % aperture horizontal points
    N = params.N;                                            % aperture vertical points
    A = params.A;                                            % image horizontal pixels
    B = params.B;                                            % image vertical pixels

    c0    = physconst('lightspeed');
    F0    = params.F0;                                       % center frequency
    z0_mm = params.z0;                                       % target range (mm)
    dx    = params.dx;                                       % aperture step (mm)
    dy    = params.dy;
    bbox  = params.bbox;                                     % [xmin xmax ymin ymax] (mm)

    % Convert to meters
    z0_m   = z0_mm * 1e-3;
    dxm    = dx   * 1e-3;
    dym    = dy   * 1e-3;
    bbox_m = bbox * 1e-3;

    % Propagation constant
    k   = 2 * pi * F0 / c0;
    cst = 1i * 2 * k;
    z2  = z0_m^2;

    % Image pixel coordinates
    wh1 = linspace(bbox_m(1), bbox_m(2), A);
    wh2 = linspace(bbox_m(3), bbox_m(4), B);

    NM = M * N;
    BA = A * B;

    H_val = complex(zeros(NM, BA));

    fprintf('    Building H matrix (%d x %d)...', NM, BA);
    tic;

    for i = 1:NM
        iy = mod(i-1, N);
        ix = (i-1-iy) / N;

        sx_i = (ix + 0.5 - M/2) * dxm;
        sy_i = (iy + 0.5 - N/2) * dym;

        for j = 1:BA
            jy = mod(j-1, B);
            jx = (j-1-jy) / B;

            px = wh1(jx+1);
            py = wh2(jy+1);

            dist2      = (sx_i - px)^2 + (sy_i - py)^2 + z2;
            H_val(i,j) = exp(cst * sqrt(dist2));
        end
    end

    fprintf(' %.3f sec\n', toc);
    H = dlarray(H_val);
end
