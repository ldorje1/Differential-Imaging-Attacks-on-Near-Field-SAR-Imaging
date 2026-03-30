clc; clear; close all;

%% ========================================================================
%  Random-Weight Attack for CSA-based Near-Field SAR/mmWave
%
%  Purpose:
%    Test whether random complex weights (without DIA optimization) can
%    still corrupt the CSA reconstruction enough to conceal the true target.
%
%  This script:
%    1) Loads victim FMCW SAR data and reconstructs a clean image.
%    2) Builds a shuffled/noise reference image (optional, for visualization).
%    3) Samples attack waveforms from X_aa, frequency-aligns them to the
%       victim range bin, and RMS-matches them to the victim measurements.
%    4) Draws random complex weights A (no optimization).
%    5) Scales A to satisfy a chosen Pa/Pr budget.
%    6) Reconstructs attacked images over multiple random trials.
%    7) Reports PSNR(A,C), SSIM(A,C), NCC(A,C), and Pa/Pr.
% ========================================================================

%% ---------------------------------------------------------------
% User settings
% ---------------------------------------------------------------
dataDir     = fullfile(pwd, 'data');
raw_dataDir = fullfile(pwd, 'raw_sar_data');

sar_algo    = 'CSA';
target_mode = 'noise';     % keep as noise reference only
rawData_select = 8;        % victim raw cube index

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

num_trials = 1;

% Sweep budgets in dB
candidate_PaPr_dB = 10;

% Concealment criterion
success_ssim_thresh = 0.001;

success_rate_thresh = 0.01;

% Random weight model
random_weight_mode = 'complex_gaussian';  % 'complex_gaussian' or 'unit_modulus'
%% ---------------------------------------------------------------
% Victim parameters
% ---------------------------------------------------------------
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

%% ---------------------------------------------------------------
% Common FMCW / SAR constants
% ---------------------------------------------------------------
c0 = physconst('lightspeed');
F0 = 77e9;
K0 = 70.295e12;
tI = 4.5225e-10;

bbox     = [-300 300 -300 300];   % CSA bbox (mm)
A_pixels = 60;
B_pixels = 60;

%% ---------------------------------------------------------------
% Load victim cube
% ---------------------------------------------------------------
sarRawData = load(fullfile(raw_dataDir, rawData(rawData_select) + ".mat")).adcDataCube;
[Nsamp, M, N] = size(sarRawData);

X_v = reshape(sarRawData, Nsamp, M * N);
Np  = M * N;

%% ---------------------------------------------------------------
% Raw data preprocessing (CSA path)
% ---------------------------------------------------------------
Echo         = permute(sarRawData, [3, 2, 1]);   % [N, M, Nsamp]
nFFTtime_v   = size(Echo, 3);
rawDataFFT_v = fft(Echo, nFFTtime_v, 3);

Ts = 1 / FS;
k0_range_bin = round(K0 * Ts * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime_v);

sarData = squeeze(rawDataFFT_v(:, :, k0_range_bin + 1)).';

for ii = 2:2:size(sarData, 1)
    sarData(ii, :) = fliplr(sarData(ii, :));
end

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

%% ---------------------------------------------------------------
% Build CSA H and linear surrogate W
% ---------------------------------------------------------------
H_csa        = dlCSA_H_matrix(params);
params.H_csa = H_csa;

params.lambda0_csa  = 1e-4;
params.p_csa        = 1.0;
params.eta_csa      = 1e-5;
params.maxIter_csa  = 100;
params.epsilon0_csa = 1e-4;

H_num_v      = double(extractdata(H_csa));
lambda_lin   = 1e-3;
HtH_v        = H_num_v' * H_num_v;
W_csa_num_v  = (HtH_v + lambda_lin * eye(size(HtH_v,1))) \ (H_num_v');
params.W_csa = dlarray(W_csa_num_v);
params.lambda_lin = lambda_lin;

%% ---------------------------------------------------------------
% Clean image
% ---------------------------------------------------------------
[~, ~, clean_img, ~, ~] = dlCSA(sarData, params);
clean_img = extractdata(clean_img);

%% ---------------------------------------------------------------
% Noise reference target image (only for visualization)
% ---------------------------------------------------------------
[rows, cols] = size(sarData);
rng(42);
sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
[~, ~, target_img, ~, ~] = dlCSA(sarData_shuffled, params);
target_img = extractdata(target_img);

%% ---------------------------------------------------------------
% Normalize images
% ---------------------------------------------------------------
global_scale = max(abs(clean_img(:))) + 1e-12;
params.global_scale = global_scale;

clean_img  = clean_img  / global_scale;
target_img = target_img / global_scale;

fprintf('clean_img  abs min/max : %.6e / %.6e\n', ...
    min(abs(clean_img(:))), max(abs(clean_img(:))));
fprintf('target_img abs min/max : %.6e / %.6e\n', ...
    min(abs(target_img(:))), max(abs(target_img(:))));

%% ---------------------------------------------------------------
% Quick visualization: clean vs shuffled noise reference
% ---------------------------------------------------------------
figure();
subplot(1,2,1);
imagesc(fliplr(clean_img)); colormap gray; colorbar;
set(gca, 'YDir', 'normal'); axis image off;
title(sprintf('Clean Image (%s)', upper(sar_algo)));

subplot(1,2,2);
imagesc(target_img); colormap gray; colorbar;
set(gca, 'YDir', 'normal'); axis image off;
title('Noise Reference');

%% ---------------------------------------------------------------
% Build attack waveform bank D
% ---------------------------------------------------------------
temp_x_aa_1 = load(fullfile(dataDir, "X_aa.mat"));   X_aa_1 = temp_x_aa_1.X_aa;
temp_x_aa_2 = load(fullfile(dataDir, "X_aa_2.mat")); X_aa_2 = temp_x_aa_2.X_aa;

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

Xspec = fft(X_a, params.nFFTtime, 1);
[~, b0] = max(abs(Xspec), [], 1);
f0    = (b0 - 1) * FS / params.nFFTtime;
f_tgt = (params.k0_range_bin) * FS / params.nFFTtime;
Delta = f0 - f_tgt;

t = (0:Nsamp-1).' / FS;
P = exp(-1j * 2*pi * (t * Delta));
D = P .* X_a;

colrms = @(X) sqrt(mean(abs(X).^2, 1));
scale  = (colrms(X_v) + eps) ./ (colrms(D) + eps);
D      = D .* scale;

Pr_ref = norm(X_v, 'fro')^2 + 1e-12;

%% ---------------------------------------------------------------
% Power sweep with random weights
% ---------------------------------------------------------------
fprintf('\n============================================================\n');
fprintf('Random-weight concealment sweep for %s on %s\n', upper(sar_algo), rawData(rawData_select));
fprintf('Trials per budget: %d\n', num_trials);
fprintf('Success criterion: SSIM(A,C) <= %.3f\n', success_ssim_thresh);
fprintf('Required success rate: %.2f\n', success_rate_thresh);
fprintf('============================================================\n\n');

all_budget_results = struct();
best_budget_found = false;
chosen_budget_idx = NaN;

for b = 1:numel(candidate_PaPr_dB)

    target_PaPr_dB = candidate_PaPr_dB(b);
    target_PaPr    = 10^(target_PaPr_dB/10);

    psnr_AC_trials = zeros(num_trials,1);
    ssim_AC_trials = zeros(num_trials,1);
    ncc_AC_trials  = zeros(num_trials,1);
    pa_pr_trials   = zeros(num_trials,1);
    mse_AC_trials  = zeros(num_trials,1);

    success_flags  = false(num_trials,1);

    adv_imgs_trial = cell(num_trials,1);
    A_store_trial  = cell(num_trials,1);

    fprintf('Testing budget Pa/Pr = %.2f dB ...\n', target_PaPr_dB);

    for tr = 1:num_trials
        rng(10000 + 100*b + tr);

        switch lower(random_weight_mode)
            case 'complex_gaussian'
                A_rand = randn(Np,1) + 1j*randn(Np,1);
            case 'unit_modulus'
                theta = 2*pi*rand(Np,1);
                A_rand = exp(1j*theta);
            otherwise
                error('Unknown random_weight_mode.');
        end

        delta_rand = D .* (A_rand.');
        Pa_rand    = norm(delta_rand, 'fro')^2;

        scale_pow = sqrt((target_PaPr * Pr_ref) / (Pa_rand + 1e-12));
        A_rand    = A_rand * scale_pow;

        [adv_img, PaPr_final, ~] = reconstruct_adv_image_random_csa(X_v, D, A_rand, params);

        [mse_AC, ncc_AC, ssim_AC, psnr_AC] = compare_to_clean(adv_img, clean_img);

        mse_AC_trials(tr) = mse_AC;
        ncc_AC_trials(tr) = ncc_AC;
        ssim_AC_trials(tr) = ssim_AC;
        psnr_AC_trials(tr) = psnr_AC;
        pa_pr_trials(tr) = PaPr_final;

        success_flags(tr) = (ssim_AC <= success_ssim_thresh);

        adv_imgs_trial{tr} = adv_img;
        A_store_trial{tr}  = A_rand;
    end

    success_rate = mean(success_flags);

    fprintf('  PSNR(A,C): %.2f ± %.2f dB\n', mean(psnr_AC_trials), std(psnr_AC_trials));
    fprintf('  SSIM(A,C): %.4f ± %.4f\n', mean(ssim_AC_trials), std(ssim_AC_trials));
    fprintf('  NCC(A,C) : %.4f ± %.4f\n', mean(ncc_AC_trials),  std(ncc_AC_trials));
    fprintf('  Pa/Pr    : %.4e ± %.4e (linear)\n', mean(pa_pr_trials), std(pa_pr_trials));
    fprintf('  Success rate (SSIM<=%.3f): %.2f\n\n', success_ssim_thresh, success_rate);

    all_budget_results(b).PaPr_dB      = target_PaPr_dB;
    all_budget_results(b).PaPr_lin     = mean(pa_pr_trials);
    all_budget_results(b).psnr_AC_mean = mean(psnr_AC_trials);
    all_budget_results(b).psnr_AC_std  = std(psnr_AC_trials);
    all_budget_results(b).ssim_AC_mean = mean(ssim_AC_trials);
    all_budget_results(b).ssim_AC_std  = std(ssim_AC_trials);
    all_budget_results(b).ncc_AC_mean  = mean(ncc_AC_trials);
    all_budget_results(b).ncc_AC_std   = std(ncc_AC_trials);
    all_budget_results(b).mse_AC_mean  = mean(mse_AC_trials);
    all_budget_results(b).mse_AC_std   = std(mse_AC_trials);
    all_budget_results(b).success_rate = success_rate;
    all_budget_results(b).adv_imgs     = adv_imgs_trial;
    all_budget_results(b).A_store      = A_store_trial;
    all_budget_results(b).success_flags = success_flags;
    all_budget_results(b).psnr_AC_trials = psnr_AC_trials;
    all_budget_results(b).ssim_AC_trials = ssim_AC_trials;
    all_budget_results(b).ncc_AC_trials  = ncc_AC_trials;
    all_budget_results(b).pa_pr_trials   = pa_pr_trials;

    if ~best_budget_found && (success_rate >= success_rate_thresh)
        best_budget_found = true;
        chosen_budget_idx = b;
    end
end

%% ---------------------------------------------------------------
% Choose operating budget
% ---------------------------------------------------------------
if best_budget_found
    fprintf('Chosen smallest successful budget: %.2f dB\n', ...
        all_budget_results(chosen_budget_idx).PaPr_dB);
else
    chosen_budget_idx = numel(all_budget_results);
    fprintf('No budget met success-rate criterion.\n');
    fprintf('Using strongest tested budget: %.2f dB\n', ...
        all_budget_results(chosen_budget_idx).PaPr_dB);
end

chosen = all_budget_results(chosen_budget_idx);

succ_idx = find(chosen.success_flags);
if ~isempty(succ_idx)
    [~, ord] = sort(chosen.ssim_AC_trials(succ_idx), 'ascend');
    pick = succ_idx(ord(max(1, round(numel(ord)/2))));
else
    [~, pick] = min(chosen.ssim_AC_trials);
end

adv_img_best = chosen.adv_imgs{pick};
A_best       = chosen.A_store{pick}; 

%% ---------------------------------------------------------------
% Final summary
% ---------------------------------------------------------------
fprintf('\n============================================================\n');
fprintf('FINAL TABLE NUMBERS for %s on %s\n', upper(sar_algo), rawData(rawData_select));
fprintf('Use budget: %.2f dB\n', chosen.PaPr_dB);
fprintf('PSNR(A,C): %.2f ± %.2f dB\n', chosen.psnr_AC_mean, chosen.psnr_AC_std);
fprintf('SSIM(A,C): %.4f ± %.4f\n', chosen.ssim_AC_mean, chosen.ssim_AC_std);
fprintf('NCC(A,C) : %.4f ± %.4f\n', chosen.ncc_AC_mean, chosen.ncc_AC_std);
fprintf('Pa/Pr    : %.4f (linear mean), %.4f dB\n', ...
    chosen.PaPr_lin, 10*log10(chosen.PaPr_lin + 1e-12));
fprintf('============================================================\n');

%% ---------------------------------------------------------------
% Budget sweep figure
% ---------------------------------------------------------------
figure;
yyaxis left;
plot([all_budget_results.PaPr_dB], [all_budget_results.ssim_AC_mean], '-o', 'LineWidth', 1.5);
ylabel('Mean SSIM(A,C)');
grid on;

yyaxis right;
plot([all_budget_results.PaPr_dB], [all_budget_results.success_rate], '-s', 'LineWidth', 1.5);
ylabel('Success rate');

xlabel('Pa/Pr (dB)');
title(sprintf('Random-weight concealment sweep: %s / %s', upper(sar_algo), rawData(rawData_select)));

%% ---------------------------------------------------------------
% Qualitative figure
% ---------------------------------------------------------------
figure;
subplot(1,3,1);
imagesc(fliplr(clean_img)); colormap gray; colorbar;
set(gca, 'YDir', 'normal'); axis image off;
title('Clean image');

subplot(1,3,2);
imagesc(adv_img_best); colormap gray; colorbar;
set(gca, 'YDir', 'normal'); axis image off;
title(sprintf('Random attack image\nPa/Pr = %.2f dB', chosen.PaPr_dB));

subplot(1,3,3);
imagesc(adv_img_best - clean_img); colormap gray; colorbar;
set(gca, 'YDir', 'normal'); axis image off;
title('Difference (A-C)');

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function [adv_img, PaPr_final, PaPr_final_dB] = reconstruct_adv_image_random_csa(X_v, D, A_rand, params)

    Y_opt  = X_v + D .* (A_rand.');
    Y_cube = reshape(Y_opt, params.Nsamp, params.M, params.N);

    Echo_att       = permute(Y_cube, [3, 2, 1]);
    rawDataFFT_att = fft(Echo_att, params.nFFTtime, 3);
    sarData_att    = squeeze(rawDataFFT_att(:, :, params.k0_range_bin + 1)).';

    for ii = 2:2:size(sarData_att, 1)
        sarData_att(ii, :) = fliplr(sarData_att(ii, :));
    end

    [~, ~, atkImage_abs, ~, ~] = dlCSA(sarData_att, params);
    adv_img = gather(extractdata(atkImage_abs));
    adv_img = adv_img / params.global_scale;

    delta_opt     = D .* (A_rand.');
    Pa_final      = norm(delta_opt, 'fro')^2;
    Pr_final      = norm(X_v, 'fro')^2 + 1e-12;
    PaPr_final    = Pa_final / Pr_final;
    PaPr_final_dB = 10 * log10(PaPr_final + 1e-12);

    PaPr_final = PaPr_final;
end

function [mse_AC, ncc_AC, ssim_AC, psnr_AC] = compare_to_clean(adv_img, clean_img)

    mse_AC = mean((adv_img(:) - clean_img(:)).^2);

    num_AC = sum(adv_img(:) .* clean_img(:));
    den_AC = sqrt(sum(adv_img(:).^2) * sum(clean_img(:).^2)) + 1e-12;
    ncc_AC = num_AC / den_AC;

    data_range_C = (max(clean_img(:)) - min(clean_img(:))) + 1e-12;

    if exist('ssim','file') == 2
        ssim_AC = ssim(adv_img, clean_img, 'DynamicRange', data_range_C);
    else
        ssim_AC = NaN;
    end

    if exist('psnr','file') == 2
        psnr_AC = psnr(adv_img, clean_img, data_range_C);
    else
        psnr_AC = 10 * log10((data_range_C^2) / (mse_AC + 1e-12));
    end
end

%% ========================================================================
% CSA FUNCTIONS
% ========================================================================

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

    [M_meas, ~] = size(H);

    temp1 = H' * H;
    HH_ys = H' * ys;

    alpha_hat_prev = HH_ys;
    alpha_hat      = alpha_hat_prev;
    r      = Inf;
    n      = 0;
    beta_n = 1;

    fprintf('Starting SBRIM (numeric)...\n');

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
    dxm    = dx * 1e-3;
    dym    = dy * 1e-3;
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