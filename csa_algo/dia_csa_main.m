clc; clear; close all;

%% ========================================================================
%  Differential Imaging Attack (DIA) for CSA-based Near-Field SAR/mmWave
%
%  End-to-end differentiable attack that optimizes complex per-aperture gains
%  A to drive the CSA reconstructor toward a target image.
%
%  The script:
%    1) Loads victim FMCW SAR data and reconstructs a clean image.
%    2) Builds a target image using either shuffled measurements ("noise")
%       or a separate object raw cube ("object").
%    3) Samples attack waveforms from X_aa, frequency-aligns them to the
%       victim range bin, and RMS-matches them to the victim measurements.
%    4) Runs gradient-based optimization on A with optional Pa/Pr and |A|
%       projections.
%    5) Reconstructs the final attacked image and reports image-domain and
%       signal-domain metrics.
% ========================================================================

%% ---------------------------------------------------------------
%  User selection: SAR imaging algorithm
% ---------------------------------------------------------------
dataDir     = fullfile(pwd, 'data');
raw_dataDir = fullfile(pwd, 'raw_sar_data');

sar_algo    = 'CSA';                  % fixed for this script
target_mode = 'noise';               % 'noise' or 'object'

target_raw_select = 1;                % which raw cube to use as OBJECT target (1..10)

rawData_select    = 6;                % victim raw cube index (1..10)

rawData = [ ...
    "knife", ...               % 1
    "plier", ...               % 2
    "scissor", ...             % 3
    "screw_driver", ...        % 4
    "sharp_paint_speader", ... % 5
    "dragger", ...             % 6
    "wrench", ...              % 7
    "gun", ...                 % 8
    "rifle", ...               % 9
    "butcher_knife" ...        % 10
];

%% ---------------------------------------------------------------
%  Step 2: Load Victim SAR Data and Setup CSA Imaging Parameters
% ---------------------------------------------------------------

% ---------- Victim parameters (geometry + sampling rate) ----------
switch lower(rawData(rawData_select))
    case "knife",                dx = 1; dy = 1; z0 = 185; FS = 5000e3;
    case "plier",                dx = 1; dy = 2; z0 = 210; FS = 5000e3;
    case "scissor",              dx = 1; dy = 2; z0 = 215; FS = 5000e3;
    case "screw_driver",         dx = 1; dy = 2; z0 = 230; FS = 5000e3;
    case "sharp_paint_speader",  dx = 1; dy = 2; z0 = 180; FS = 5000e3;
    case "dragger",              dx = 1; dy = 2; z0 = 195; FS = 5000e3;
    case "wrench",               dx = 1; dy = 1; z0 = 170; FS = 9121e3;
    case "gun",                  dx = 1; dy = 1; z0 = 185; FS = 9121e3;
    case "rifle",                dx = 1; dy = 1; z0 = 185; FS = 9121e3;
    case "butcher_knife",        dx = 1; dy = 2; z0 = 210; FS = 5000e3;
    otherwise
        error("Unknown rawData selection: %s", rawData(rawData_select));
end

% ---------- Target parameters (only if object mode) ----------
if strcmpi(target_mode, 'object')
    switch lower(rawData(target_raw_select))
        case "knife",                dx_t = 1; dy_t = 1; z0_tgt = 185; FS_tgt = 5000e3;
        case "plier",                dx_t = 1; dy_t = 2; z0_tgt = 220; FS_tgt = 5000e3;
        case "scissor",              dx_t = 1; dy_t = 2; z0_tgt = 215; FS_tgt = 5000e3;
        case "screw_driver",         dx_t = 1; dy_t = 2; z0_tgt = 230; FS_tgt = 5000e3;
        case "sharp_paint_speader",  dx_t = 1; dy_t = 2; z0_tgt = 180; FS_tgt = 5000e3;
        case "dragger",              dx_t = 1; dy_t = 2; z0_tgt = 195; FS_tgt = 5000e3;
        case "wrench",               dx_t = 1; dy_t = 1; z0_tgt = 180; FS_tgt = 9121e3;
        case "gun",                  dx_t = 1; dy_t = 1; z0_tgt = 185; FS_tgt = 9121e3;
        case "rifle",                dx_t = 1; dy_t = 1; z0_tgt = 185; FS_tgt = 9121e3;
        case "butcher_knife",        dx_t = 1; dy_t = 2; z0_tgt = 210; FS_tgt = 5000e3;
        otherwise
            error("Unknown target_raw_select: %s", rawData(target_raw_select));
    end
end

% ---------- Common FMCW / SAR constants ----------
c0 = physconst('lightspeed');
F0 = 77e9;
K0 = 70.295e12;
tI = 4.5225e-10;

bbox = [-300 300 -300 300];          % CSA bbox (mm)

% Image grid size used by CSA
A_pixels = 60;
B_pixels = 60;

% ---------- Load victim cube ----------
sarRawData = load(fullfile(raw_dataDir, rawData(rawData_select) + ".mat")).adcDataCube;
if strcmpi(target_mode,'object')
    sarRawData_tgt = load(fullfile(raw_dataDir, rawData(target_raw_select) + ".mat")).adcDataCube;
end

[Nsamp, M, N] = size(sarRawData);

X_v = reshape(sarRawData, Nsamp, M * N);      % Nsamp x Np
Np  = M * N;

% ---------------------------------------------------------------
%  Raw data pre-processing (CSA path)
% ---------------------------------------------------------------
Echo         = permute(sarRawData, [3, 2, 1]);             % [N, M, Nsamp]
nFFTtime_v   = size(Echo, 3);
rawDataFFT_v = fft(Echo, nFFTtime_v, 3);

Ts = 1 / FS;
k0_range_bin = round(K0 * Ts * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime_v);

% Gate chosen bin and transpose to M x N style used below
sarData = squeeze(rawDataFFT_v(:, :, k0_range_bin + 1)).';

% Serpentine correction
for ii = 2:2:size(sarData, 1)
    sarData(ii, :) = fliplr(sarData(ii, :));
end

% Package parameters used by the imaging operator
params = struct( ...
    'z0',           z0, ...
    'dx',           dx, ...
    'dy',           dy, ...
    'bbox',         bbox, ...
    'Nsamp',        Nsamp, ...
    'nFFTtime',     nFFTtime_v, ...
    'N',            N, ...
    'M',            M, ...
    'A',            A_pixels, ...
    'B',            B_pixels, ...
    'F0',           F0, ...
    'k0_range_bin', k0_range_bin, ...
    'sar_algo',     sar_algo, ...
    'target_mode',  target_mode);

% Build victim CSA H and linear surrogate W
H_csa         = dlCSA_H_matrix(params);
params.H_csa  = H_csa;

params.lambda0_csa  = 1e-4;
params.p_csa        = 1.0;
params.eta_csa      = 1e-5;
params.maxIter_csa  = 100;
params.epsilon0_csa = 1e-4;

H_num_v      = double(extractdata(H_csa));
lambda_lin   = 1e-3;
HtH_v        = H_num_v' * H_num_v;
W_csa_num_v  = (HtH_v + lambda_lin * eye(size(HtH_v,1))) \ (H_num_v');
params.W_csa      = dlarray(W_csa_num_v);
params.lambda_lin = lambda_lin;

% Generate clean image
[~, ~, clean_img, ~, ~] = dlCSA(sarData, params);
clean_img = extractdata(clean_img);

% ---------------------------------------------------------------
% Generate target image
% ---------------------------------------------------------------
if strcmpi(target_mode,'noise')
    [rows, cols] = size(sarData);
    rng(42);
    sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);

    [~, ~, target_img, ~, ~] = dlCSA(sarData_shuffled, params);
    target_img = extractdata(target_img);

elseif strcmpi(target_mode,'object')
    [Nsamp_t, M_t, N_t] = size(sarRawData_tgt);

    Echo_t       = permute(sarRawData_tgt, [3, 2, 1]);
    nFFTtime_t   = size(Echo_t, 3);
    rawDataFFT_t = fft(Echo_t, nFFTtime_t, 3);

    Ts_t  = 1 / FS_tgt;
    kbin_t = round(K0 * Ts_t * (2 * z0_tgt * 1e-3 / c0 + tI) * nFFTtime_t);

    sarData_t = squeeze(rawDataFFT_t(:, :, kbin_t + 1)).';

    for ii = 2:2:size(sarData_t, 1)
        sarData_t(ii, :) = fliplr(sarData_t(ii, :));
    end

    params_t = params;
    params_t.z0           = z0_tgt;
    params_t.dx           = dx_t;
    params_t.dy           = dy_t;
    params_t.Nsamp        = Nsamp_t;
    params_t.M            = M_t;
    params_t.N            = N_t;
    params_t.nFFTtime     = nFFTtime_t;
    params_t.k0_range_bin = kbin_t;

    H_t        = dlCSA_H_matrix(params_t);
    params_t.H_csa = H_t;

    H_num_t     = double(extractdata(H_t));
    HtH_t       = H_num_t' * H_num_t;
    W_csa_num_t = (HtH_t + lambda_lin * eye(size(HtH_t,1))) \ (H_num_t');
    params_t.W_csa = dlarray(W_csa_num_t);

    [~, ~, target_img, ~, ~] = dlCSA(sarData_t, params_t);
    target_img = extractdata(target_img);

else
    error("target_mode must be 'noise' or 'object'.");
end

% --- Make target image match victim image grid size if needed
if ~isequal(size(target_img), size(clean_img))
    fprintf('Resizing target_img %s -> %s to match clean_img grid.\n', ...
        mat2str(size(target_img)), mat2str(size(clean_img)));

    if exist('imresize','file') == 2
        target_img = imresize(target_img, size(clean_img), 'bilinear');
    else
        [X, Y]  = meshgrid(linspace(0,1,size(target_img,2)), linspace(0,1,size(target_img,1)));
        [Xq,Yq] = meshgrid(linspace(0,1,size(clean_img,2)),  linspace(0,1,size(clean_img,1)));
        target_img = interp2(X, Y, target_img, Xq, Yq, 'linear', 0);
    end
end

% ---------------------------------------------------------------
% Quick visualization: clean vs target
% ---------------------------------------------------------------
figure();
subplot(1,2,1);
imagesc(clean_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
title(sprintf('Clean Image (%s)', upper(sar_algo)));
axis image off;

subplot(1,2,2);
imagesc(target_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
title(sprintf('Target Image (%s)', upper(sar_algo)));
axis image off;

fprintf('clean_img  abs min/max : %.6e / %.6e\n', ...
    min(abs(clean_img(:))), max(abs(clean_img(:))));
fprintf('target_img abs min/max : %.6e / %.6e\n', ...
    min(abs(target_img(:))), max(abs(target_img(:))));

% ---------------------------------------------------------------
% Normalization + store dlarray versions in params
% ---------------------------------------------------------------
global_scale = max(abs(clean_img(:))) + 1e-12;
params.global_scale = global_scale;

clean_img  = clean_img  / global_scale;
target_img = target_img / global_scale;

params.clean_img  = dlarray(clean_img,  "SS");
params.target_img = dlarray(target_img, "SS");

fprintf('clean_img  abs min/max : %.6e / %.6e\n', ...
    min(abs(clean_img(:))), max(abs(clean_img(:))));
fprintf('target_img abs min/max : %.6e / %.6e\n', ...
    min(abs(target_img(:))), max(abs(target_img(:))));

%% ---------------------------------------------------------------
%  Step 3: Build Attack Waveforms and Optimize Complex Gains A
% ---------------------------------------------------------------

% Step 1: Load attack signal pool (X_aa)
temp_x_aa_1 = load(fullfile(dataDir, "X_aa.mat"));   X_aa_1 = temp_x_aa_1.X_aa;
temp_x_aa_2 = load(fullfile(dataDir, "X_aa_2.mat")); X_aa_2 = temp_x_aa_2.X_aa;

% ---------------------------------------------------------------
% 3.1 Select and frequency-align attack waveforms
% ---------------------------------------------------------------
if Nsamp == 512 && M*N > 40000
    X_aa_temp = [X_aa_1; X_aa_2];
    X_aa      = [X_aa_temp, X_aa_temp];
elseif Nsamp == 512 && M*N == 40000
    X_aa = [X_aa_1; X_aa_2];
else
    X_aa = X_aa_1;
end

targetK = M * N;
rng(0);

sample_idx = randi(size(X_aa,2), [1, targetK]);
X_a_pool   = X_aa(:, sample_idx);

sel_idx = randi(size(X_a_pool, 2), [1, Np]);
X_a     = X_a_pool(:, sel_idx);

% ---------------------------------------------------------------
% Per-column frequency shift to victim range bin
% ---------------------------------------------------------------
Xspec   = fft(X_a, params.nFFTtime, 1);
[~, b0] = max(abs(Xspec), [], 1);
f0      = (b0 - 1) * FS / params.nFFTtime;
f_tgt   = (params.k0_range_bin) * FS / params.nFFTtime;
Delta   = f0 - f_tgt;

t = (0:Nsamp-1).' / FS;
P = exp(-1j * 2*pi * (t * Delta));
D = P .* X_a;

% ---------------------------------------------------------------
% Scale D to match victim RMS per column
% ---------------------------------------------------------------
colrms = @(X) sqrt(mean(abs(X).^2, 1));
scale  = (colrms(X_v) + eps) ./ (colrms(D) + eps);
D      = D .* scale;

%% ---------------------------------------------------------------
% DIA Optimization
% ---------------------------------------------------------------
save_final_images = false;
save_results_data = false;

maxIter   = 500;
lr_re     = 1e3;
lr_im     = 1e3;
lambda_L2 = 1e-5;

% -----------------------------------------------------------------
% Constraint options
% -----------------------------------------------------------------
use_pa_pr_projection = true;
PaPr_max_dB          = -10;
PaPr_max             = 10^(PaPr_max_dB/10);

use_amax_projection  = false;
Amax                 = 2;

Pr_ref = norm(X_v, 'fro')^2 + 1e-12;

A_re = dlarray(1e-3 * randn(Np, 1, 'double'));
A_im = dlarray(1e-3 * randn(Np, 1, 'double'));

loss_hist     = zeros(maxIter, 1);
meanA_hist    = zeros(maxIter, 1);
maxA_hist     = zeros(maxIter, 1);
PaPr_hist     = zeros(maxIter, 1);
PaPrdB_hist   = zeros(maxIter, 1);

fprintf(['\nOptimizing complex gain A for all locations (Np=%d) with SAR algorithm: %s ' ...
         '| lr_re=%.3g, lr_im=%.3g | lambda_L2=%.3g | ' ...
         'Pa/Pr<=%.2f dB (%d) | |A|<=%.3g (%d)\n'], ...
        Np, params.sar_algo, lr_re, lr_im, lambda_L2, ...
        PaPr_max_dB, use_pa_pr_projection, Amax, use_amax_projection);

% ---------------------------------------------------------------
% Main optimization loop (reference-style diminishing returns)
% ---------------------------------------------------------------
maxIter_cap = maxIter;

check_every   = 25;
W             = 50;
rel_drop_min  = 1e-2;
abs_drop_min  = 0;
K_weak        = 3;
weakWinCount  = 0;

bestLoss = inf;
bestIter = 0;

bestA_re = A_re;
bestA_im = A_im;

iter = 0;

while true
    if iter >= maxIter_cap
        fprintf('Reached maxIter cap (%d).\n', maxIter_cap);
        break;
    end

    iter = iter + 1;

    % ---- Forward/grad through full pipeline
    [loss, gRe, gIm, ~] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);

    lossVal         = double(gather(extractdata(loss)));
    loss_hist(iter) = lossVal;

    % ---- Gradient step
    A_re = A_re - lr_re * gRe;
    A_im = A_im - lr_im * gIm;

    % ---- Numeric complex A for projections/monitoring
    A_num = extractdata(A_re) + 1j * extractdata(A_im);

    % -----------------------------------------------------------
    % Optional projection 1: |A| <= Amax
    % -----------------------------------------------------------
    if use_amax_projection
        mags = abs(A_num);
        over = mags > Amax;
        if any(over)
            scale_amax       = ones(size(mags));
            scale_amax(over) = Amax ./ mags(over);
            A_num            = A_num .* scale_amax;
        end
    end

    % -----------------------------------------------------------
    % Projection 2: enforce Pa/Pr <= PaPr_max
    % -----------------------------------------------------------
    delta_now = D .* (A_num.');
    Pa_now    = norm(delta_now, 'fro')^2;
    PaPr_now  = Pa_now / Pr_ref;

    if use_pa_pr_projection && (PaPr_now > PaPr_max)
        scale_pow = sqrt(PaPr_max / (PaPr_now + 1e-12));
        A_num     = A_num * scale_pow;

        delta_now = D .* (A_num.');
        Pa_now    = norm(delta_now, 'fro')^2;
        PaPr_now  = Pa_now / Pr_ref;
    end
    PaPr_now_dB = 10 * log10(PaPr_now + 1e-12);

    % ---- Write back projected A
    A_re = dlarray(real(A_num));
    A_im = dlarray(imag(A_num));

    % ---- Track stats
    meanA_now = mean(abs(A_num), 'all');
    maxA_now  = max(abs(A_num), [], 'all');

    meanA_hist(iter)  = meanA_now;
    maxA_hist(iter)   = maxA_now;
    PaPr_hist(iter)   = PaPr_now;
    PaPrdB_hist(iter) = PaPr_now_dB;

    % -----------------------------------------------------------
    % Update best
    % -----------------------------------------------------------
    if lossVal < bestLoss
        bestLoss = lossVal;
        bestIter = iter;
        bestA_re = A_re;
        bestA_im = A_im;
    end

    % -----------------------------------------------------------
    % Logging
    % -----------------------------------------------------------
    if mod(iter, 25) == 0 || iter == 1 || iter == maxIter_cap
        [loss_log, ~, ~, atkImage_log] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);
        lossVal_log = double(gather(extractdata(loss_log)));

        mse_AT = double(gather(extractdata(mean((atkImage_log - params.target_img).^2, 'all'))));
        mse_CA = double(gather(extractdata(mean((atkImage_log - params.clean_img ).^2, 'all'))));

        gRe_num = extractdata(gRe);
        gIm_num = extractdata(gIm);
        Gnow    = max([max(abs(gRe_num), [], 'all'), max(abs(gIm_num), [], 'all')]);

        fprintf(['Iter %04d | Loss=%.4e (best=%.4e @%d) | G=%.6e | ' ...
                 'MSE(A,T)=%.4e, MSE(C,A)=%.4e | E|A|=%.4e, max|A|=%.4e | ' ...
                 'Pa/Pr=%.4e (%.2f dB)\n'], ...
                iter, lossVal_log, bestLoss, bestIter, Gnow, ...
                mse_AT, mse_CA, meanA_now, maxA_now, PaPr_now, PaPr_now_dB);
    end

    % -----------------------------------------------------------
    % Early stop rule: diminishing returns over a window
    % -----------------------------------------------------------
    if iter > W && mod(iter, check_every) == 0
        L_old = loss_hist(iter - W);
        L_new = loss_hist(iter);

        rel_drop = (L_old - L_new) / max(abs(L_old), 1e-12);
        abs_drop = (L_old - L_new);

        if abs_drop_min > 0
            isWeak = (rel_drop < rel_drop_min) && (abs_drop < abs_drop_min);
        else
            isWeak = (rel_drop < rel_drop_min);
        end

        if isWeak
            weakWinCount = weakWinCount + 1;
        else
            weakWinCount = 0;
        end

        if weakWinCount >= K_weak
            fprintf(['Early stop: diminishing returns. Over last %d iters: ' ...
                     'rel_drop=%.3e, abs_drop=%.3e. Triggered %d/%d.\n'], ...
                     W, rel_drop, abs_drop, weakWinCount, K_weak);
            break;
        end
    end
end

% ---- Trim history arrays to actual iters
loss_hist   = loss_hist(1:iter);
meanA_hist  = meanA_hist(1:iter);
maxA_hist   = maxA_hist(1:iter);
PaPr_hist   = PaPr_hist(1:iter);
PaPrdB_hist = PaPrdB_hist(1:iter);

% ---- Restore best A
A_re = bestA_re;
A_im = bestA_im;

fprintf('-----------------------------\n');
fprintf('Done. Best loss %.4e at iter %d (ran %d iters).\n', bestLoss, bestIter, iter);
fprintf('-----------------------------\n');

% ---------------------------------------------------------------
%  Step 3.3: Reconstruct Attacked Image with Optimal A
% ---------------------------------------------------------------

A_opt = extractdata(A_re) + 1j * extractdata(A_im);

Y_opt = X_v + D .* (A_opt.');
Y_cube = reshape(Y_opt, Nsamp, M, N);

Echo_att       = permute(Y_cube, [3, 2, 1]);
rawDataFFT_att = fft(Echo_att, params.nFFTtime, 3);
sarData_att    = squeeze(rawDataFFT_att(:, :, params.k0_range_bin + 1)).';

for ii = 2:2:size(sarData_att, 1)
    sarData_att(ii, :) = fliplr(sarData_att(ii, :));
end

[~, ~, atkImage_abs, ~, ~] = dlCSA(sarData_att, params);
adv_img = gather(extractdata(atkImage_abs));
adv_img = adv_img / params.global_scale;

delta_opt     = D .* (A_opt.');
Pa_final      = norm(delta_opt, 'fro')^2;
Pr_final      = norm(X_v, 'fro')^2 + 1e-12;
PaPr_final    = Pa_final / Pr_final;
PaPr_final_dB = 10 * log10(PaPr_final + 1e-12);

% ---------------------------------------------------------------
% Final metric evaluation
% ---------------------------------------------------------------
visio_fmt = 'png';

clean_img  = double(clean_img);
target_img = double(target_img);
adv_img    = double(adv_img);

mse_AT = mean((adv_img(:) - target_img(:)).^2);
mse_AC = mean((adv_img(:) - clean_img(:)).^2);

num_AT = sum(adv_img(:) .* target_img(:));
den_AT = sqrt(sum(adv_img(:).^2) * sum(target_img(:).^2)) + 1e-12;
ncc_AT = num_AT / den_AT;

num_AC = sum(adv_img(:) .* clean_img(:));
den_AC = sqrt(sum(adv_img(:).^2) * sum(clean_img(:).^2)) + 1e-12;
ncc_AC = num_AC / den_AC;

data_range_T = (max(target_img(:)) - min(target_img(:))) + 1e-12;
data_range_C = (max(clean_img(:))  - min(clean_img(:)))  + 1e-12;

if exist('ssim','file') == 2
    ssim_AT = ssim(adv_img, target_img, 'DynamicRange', data_range_T);
    ssim_AC = ssim(adv_img, clean_img,  'DynamicRange', data_range_C);
else
    ssim_AT = NaN;
    ssim_AC = NaN;
end

if exist('psnr','file') == 2
    psnr_AT = psnr(adv_img, target_img, data_range_T);
    psnr_AC = psnr(adv_img, clean_img,  data_range_C);
else
    psnr_AT = 10 * log10((data_range_T^2) / (mse_AT + 1e-12));
    psnr_AC = 10 * log10((data_range_C^2) / (mse_AC + 1e-12));
end

fprintf('\n--- %s SR-level attack metrics ---\n', upper(params.sar_algo));
fprintf('MSE(A,T)      : %.4e\n', mse_AT);
fprintf('MSE(A,C)      : %.4e\n', mse_AC);
fprintf('NCC(A,T)      : %.4f\n', ncc_AT);
fprintf('NCC(A,C)      : %.4f\n', ncc_AC);
fprintf('PSNR(A,C)     : %.2f dB\n', psnr_AC);
fprintf('SSIM(A,C)     : %.4f\n', ssim_AC);
fprintf('PSNR(A,T)     : %.2f dB\n', psnr_AT);
fprintf('SSIM(A,T)     : %.4f\n', ssim_AT);
fprintf('Pa/Pr         : %.4e (%.2f dB)\n', PaPr_final, PaPr_final_dB);

% ---------------------------------------------------------------
% Visualization: clean, target, attacked, and difference
% ---------------------------------------------------------------
figure();

subplot(2,2,1);
imagesc(clean_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title(sprintf('Clean %s output image', upper(params.sar_algo)));

subplot(2,2,2);
imagesc(target_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Target image (desired attacked)');

subplot(2,2,3);
imagesc(adv_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Adversarial image (attacked)');

subplot(2,2,4);
imagesc(adv_img - clean_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Diff (A-C)');

%% ---------------------------------------------------------------
% Option to save individual images
% ---------------------------------------------------------------
if use_pa_pr_projection
    constraint_folder = 'power_constraint';
else
    constraint_folder = 'no_power_constraint';
end

victim_obj      = char(rawData(rawData_select));
algo_lower      = lower(params.sar_algo);
algo_obj_folder = sprintf('%s_%s', algo_lower, victim_obj);

if strcmpi(target_mode, 'noise')
    target_suffix = 'tgt_noise';
    target_obj_for_filename = 'noise';
else
    target_obj = char(rawData(target_raw_select));
    target_suffix = sprintf('tgt_%s', target_obj);
    target_obj_for_filename = target_obj;
end

if save_final_images
    out_dir = fullfile('figures', constraint_folder, algo_obj_folder);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    img_sets = {
        'clean',  clean_img,           sprintf('%s_%s', victim_obj, target_suffix);
        'target', target_img,          target_obj_for_filename;
        'adv',    adv_img,             sprintf('%s_%s', victim_obj, target_suffix);
        'diff',   adv_img - clean_img, sprintf('%s_%s', victim_obj, target_suffix)
    };

    for i = 1:size(img_sets, 1)
        prefix    = img_sets{i,1};
        img_data  = img_sets{i,2};
        name_part = img_sets{i,3};

        out_file = fullfile(out_dir, sprintf('%s_%s_%s.%s', ...
            prefix, algo_lower, name_part, lower(visio_fmt)));

        f = figure('Visible', 'off', 'Color', 'w');
        imagesc(img_data);
        colormap gray;
        colorbar;
        set(gca, 'YDir', 'normal');
        axis image off;

        switch lower(visio_fmt)
            case 'svg'
                try
                    exportgraphics(f, out_file, 'ContentType', 'vector');
                catch
                    print(f, out_file, '-dsvg');
                end

            case 'emf'
                if ispc
                    try
                        exportgraphics(f, out_file, 'ContentType', 'vector');
                    catch
                        print(f, out_file, '-dmeta');
                    end
                else
                    warning('EMF export is mainly supported on Windows. Saving PDF instead.');
                    out_file = fullfile(out_dir, sprintf('%s_%s_%s.pdf', ...
                        prefix, algo_lower, name_part));
                    exportgraphics(f, out_file, 'ContentType', 'vector');
                end

            case 'pdf'
                exportgraphics(f, out_file, 'ContentType', 'vector');

            case 'png'
                exportgraphics(f, out_file, 'Resolution', 300);

            otherwise
                close(f);
                error('Unsupported visio_fmt: %s. Use ''svg'', ''emf'', ''pdf'', or ''png''.', visio_fmt);
        end

        close(f);
    end

    fprintf('\nIndividual images saved to "%s"\n', out_dir);
end

%% ========================================================================
%  Helper Functions
% ========================================================================

function [loss, gradRe, gradIm, atkImage] = loss_and_grad(X_v, D, A_re, A_im, params, lambda_L2)

    if ~isa(X_v, 'dlarray'), X_v = dlarray(X_v); end
    if ~isa(D,   'dlarray'), D   = dlarray(D);   end
    if nargin < 6 || isempty(lambda_L2), lambda_L2 = 0; end

    A = A_re + 1j * A_im;            % Np x 1
    Y = X_v + D .* A.';              % Nsamp x Np

    Y_cube = reshape(Y, params.Nsamp, params.M, params.N);

    EchoY      = permute(Y_cube, [3, 2, 1]);                        % [N x M x Nsamp]
    rawDataFFT = fft(EchoY, params.nFFTtime, 3);                    % FFT along fast-time
    sarData    = squeeze(rawDataFFT(:, :, params.k0_range_bin+1)).'; % [M x N]

    for ii = 2:2:size(sarData, 1)
        sarData(ii, :) = fliplr(sarData(ii, :));
    end

    ys = reshape(sarData, [], 1);                                   % (M*N x 1)

    alpha_hat_vec    = params.W_csa * ys;                           % (A*B x 1)
    B                = params.B;
    A_sz             = params.A;
    atkImage_complex = reshape(alpha_hat_vec, B, A_sz);

    atkImage = abs(atkImage_complex);
    atkImage = atkImage / params.global_scale;

    loss_im = mean((atkImage - params.target_img).^2, 'all');
    reg     = lambda_L2 * mean(abs(A).^2, 'all');
    loss    = loss_im + reg;

    [gradRe, gradIm] = dlgradient(loss, A_re, A_im);
end

function [xRangeT, yRangeT, trueImage_abs, trueImage_complx, alpha_hat_dl] = dlCSA(sarData, params)

    if isa(sarData, 'dlarray')
        sarData_num = double(extractdata(sarData));
    else
        sarData_num = double(sarData);
    end

    if isa(params.H_csa, 'dlarray')
        H_num = double(extractdata(params.H_csa));
    else
        H_num = double(params.H_csa);
    end

    ys_num = sarData_num(:);

    alpha_hat_num = CSA_SBRIM_numeric(ys_num, H_num, ...
                                      params.lambda0_csa, ...
                                      params.p_csa, ...
                                      params.eta_csa, ...
                                      params.maxIter_csa, ...
                                      params.epsilon0_csa);

    B = params.B;
    A = params.A;
    alpha_img_num = reshape(alpha_hat_num, B, A);

    trueImage_complx = dlarray(alpha_img_num);
    img_mag          = abs(alpha_img_num);
    trueImage_abs    = dlarray(img_mag, 'SS');

    alpha_hat_dl = dlarray(alpha_hat_num);

    xRangeT = params.bbox(1) + (0:A-1) * params.dx;
    yRangeT = params.bbox(3) + (0:B-1) * params.dy;
end

function alpha_hat = CSA_SBRIM_numeric(ys, H, lambda0, p, eta, maxIter, epsilon0)

    ys = double(ys);
    H  = double(H);

    [M_meas, N_pixels] = size(H);

    temp1 = H' * H;
    HH_ys = H' * ys;

    alpha_hat_prev = HH_ys;
    alpha_hat      = alpha_hat_prev;
    r      = Inf;
    n      = 0;
    beta_n = 1;

    fprintf('Starting SBRIM (numeric) with N_pixels=%d, p=%.2f...\n', N_pixels, p);

    while (r >= epsilon0) && (n < maxIter)
        n = n + 1;
        alpha_hat_prev = alpha_hat;

        alpha_sq_plus_eta = abs(alpha_hat_prev).^2 + eta;
        lambda_diag       = (p / 2) * (alpha_sq_plus_eta).^(p/2 - 1);

        Lambda_n = diag(lambda_diag);

        A_mat = temp1 + lambda0 * beta_n * Lambda_n;
        alpha_hat = A_mat \ HH_ys;

        residual = ys - H * alpha_hat;
        beta_n   = sum(abs(residual).^2) / M_meas;

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

    Ny = params.M;
    Nx = params.N;
    A  = params.A;
    B  = params.B;

    c0    = physconst('lightspeed');
    F0    = params.F0;
    z0_mm = params.z0;
    dx    = params.dx;
    dy    = params.dy;
    bbox  = params.bbox;

    z0_m   = z0_mm * 1e-3;
    dxm    = dx   * 1e-3;
    dym    = dy   * 1e-3;
    bbox_m = bbox * 1e-3;

    k   = 2 * pi * F0 / c0;
    cst = 1i * 2 * k;
    z2  = z0_m^2;

    wh1 = linspace(bbox_m(1), bbox_m(2), A);
    wh2 = linspace(bbox_m(3), bbox_m(4), B);

    NM = Ny * Nx;
    BA = A * B;

    H_val = complex(zeros(NM, BA));

    fprintf('    Building H matrix (%d x %d)...', NM, BA);
    tic;
    for i = 1:NM
        iy = mod(i-1, Ny);
        ix = (i-1-iy) / Ny;

        sx_i = (ix + 0.5 - Nx/2) * dxm;
        sy_i = (iy + 0.5 - Ny/2) * dym;

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