clc; clear; close all;

%% ========================================================================
%  Random-Weight Signal-Domain Attack for Near-Field SAR/mmWave Imaging
%
%  Purpose:
%    Demonstrate that even WITHOUT DIA optimization, random complex weights
%    can still corrupt the measurement-domain data enough to conceal the
%    true target in the reconstructed image.
%
%  This script:
%    1) Loads victim FMCW SAR data and reconstructs a clean image.
%    2) Builds a physically grounded attack waveform bank D from X_aa.
%    3) Draws random complex weights A (no gradient optimization).
%    4) Scales A to satisfy a desired Pa/Pr budget.
%    5) Reconstructs attacked images over multiple random trials.
%    6) Reports PSNR(A,C), SSIM(A,C), NCC(A,C), Pa/Pr.
%    7) Optionally searches for the smallest Pa/Pr that still conceals.
%
%  NOTE:
%    Keep the helper functions from your original script:
%      - dlMFA, dlRMA, dlBPA, dlLIA, dlBPA_H_matrix, refMF

%% ========================================================================

%% ---------------------------------------------------------------
% User settings
% ---------------------------------------------------------------
dataDir     = fullfile(pwd, 'data');
raw_dataDir = fullfile(pwd, 'raw_sar_data');

sar_algo    = 'LIA';          % 'MFA', 'RMA', 'BPA', 'LIA'

rawData_select = 10;           % victim object index

target_mode = 'noise';        % keep as noise for random conceal experiment


% Object list
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

% Number of random trials per power budget
num_trials = 1;

% ---- power sweep ----
% The script will test these budgets and pick the SMALLEST budget that
% achieves the concealment criterion often enough.
%candidate_PaPr_dB = [-20 -15 -12 -10 -8 -6 -4 -3 -2 -1 0];
%candidate_PaPr_dB = [-10 -8 -6 -4 -3 -2 -1 0];
%candidate_PaPr_dB = [-1 0 1 2 3 20];

candidate_PaPr_dB = 10;

% Concealment criterion: attacked image must be dissimilar to clean image
% You can tune this. Smaller SSIM_AC means stronger concealment/randomization.
success_ssim_thresh = 0.001;


% Require at least this fraction of trials to succeed at that budget
success_rate_thresh = 0.01;

% Random weight model:
%   'complex_gaussian' : A ~ CN(0,1)
%   'unit_modulus'     : random phase only, |A| = 1 before power scaling
random_weight_mode = 'complex_gaussian';

% Whether to show all trial images for best budget (false = show best only)
show_extra_examples = false;

% ---------------------------------------------------------------
% Victim geometry / sampling parameters
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

c0 = physconst('lightspeed');
F0 = 77e9;
K0 = 70.295e12;
tI = 4.5225e-10;

nFFTtime  = 1024;
nFFTspace = 1024;

% ---------------------------------------------------------------
% Load victim SAR data
% ---------------------------------------------------------------
sarRawData = load(fullfile(raw_dataDir, rawData(rawData_select) + ".mat")).adcDataCube;
[Nsamp, M, N] = size(sarRawData);

X_v = reshape(sarRawData, Nsamp, M * N);   % Nsamp x Np
Np  = M * N;

% ---------------------------------------------------------------
% Build clean image and noise target image
% ---------------------------------------------------------------
switch upper(sar_algo)

    case 'MFA'
        bbox = [-200 200 -200 200];

        k0_range_bin = round(K0 / FS * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);
        rawDataFFT = fft(sarRawData, nFFTtime);
        sarData    = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, ...
                        'sar_algo', sar_algo, 'target_mode', target_mode);

        [~, ~, clean_img, ~] = dlMFA(sarData, params);
        clean_img = extractdata(clean_img);

        [rows, cols] = size(sarData);
        rng(42);
        sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
        [~, ~, target_img, ~] = dlMFA(sarData_shuffled, params);
        target_img = extractdata(target_img);

    case 'RMA'
        Echo = permute(sarRawData, [3, 2, 1]);   % [N, M, Nsamp]
        bbox = [-200 200 -200 200];

        num_sample = size(Echo, 3);
        nFFTtime   = num_sample;
        rawDataFFT = fft(Echo, nFFTtime, 3);

        E = squeeze(sum(sum(abs(rawDataFFT).^2, 1), 2));
        [~, k0_range_bin] = max(E);

        label = lower(rawData(rawData_select));
        switch label
            case {"screw_driver", "gun", "rifle"}
                k0_range_bin = 8;
            case "wrench"
                k0_range_bin = 7;
        end

        sarData = squeeze(rawDataFFT(:, :, k0_range_bin)).';
        z0_t = (c0/2) * (((k0_range_bin - 1) / (K0*(1/FS)*nFFTtime)) - tI);

        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0_t, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, ...
                        'sar_algo', sar_algo, 'target_mode', target_mode);

        [~, ~, clean_img, ~] = dlRMA(sarData, params);
        clean_img = extractdata(clean_img);

        [rows, cols] = size(sarData);
        rng(42);
        sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
        [~, ~, target_img, ~] = dlRMA(sarData_shuffled, params);
        target_img = extractdata(target_img);

    case 'BPA'
        bbox = [-200 200 -200 200];

        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * (1 / FS) * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        label = lower(rawData(rawData_select));
        switch label
            case "gun"
                k0_range_bin = 14;
            case "rifle"
                k0_range_bin = 13;
        end

        sarData = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        A = 50;
        B = 50;

        params = struct('z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'Nsamp', Nsamp, 'nFFTtime', nFFTtime, ...
                        'N', N, 'M', M, 'A_bpa', A, 'B_bpa', B, ...
                        'F0', F0, 'k0_range_bin', k0_range_bin, ...
                        'sar_algo', sar_algo, 'target_mode', target_mode);

        H_bpa = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        [~, ~, clean_img, ~] = dlBPA(sarData, params, H_bpa);
        clean_img = extractdata(clean_img);

        [rows, cols] = size(sarData);
        rng(42);
        sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
        [~, ~, target_img, ~] = dlBPA(sarData_shuffled, params, H_bpa);
        target_img = extractdata(target_img);

    case 'LIA'
        bbox = [-200 200 -200 200];

        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * (1/FS) * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        label = lower(rawData(rawData_select));
        switch label
            case "gun"
                k0_range_bin = 14;
            case "rifle"
                k0_range_bin = 13;
        end

        sarData = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        A = 50;
        B = 50;

        params = struct('z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'Nsamp', Nsamp, 'nFFTtime', nFFTtime, ...
                        'N', N, 'M', M, 'A_bpa', A, 'B_bpa', B, ...
                        'F0', F0, 'k0_range_bin', k0_range_bin, ...
                        'sar_algo', sar_algo, 'target_mode', target_mode);

        H_bpa = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        NM       = M * N;
        kk       = min(40000, NM);
        rng(1000);
        params.py = sort(randperm(NM, kk));

        [~, ~, clean_img, ~] = dlLIA(sarData, params, H_bpa);
        clean_img = extractdata(clean_img);

        [~, ~, clean_img_2, ~] = dlBPA(sarData, params, H_bpa);
        clean_img_2 = extractdata(clean_img_2);

        [rows, cols] = size(sarData);
        rng(42);
        sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
        [~, ~, target_img, ~] = dlBPA(sarData_shuffled, params, H_bpa);
        target_img = extractdata(target_img);

    otherwise
        error('Invalid SAR algorithm selection.');
end

% ---------------------------------------------------------------
% Normalize images
% ---------------------------------------------------------------
if strcmpi(sar_algo, 'LIA')
    global_scale_lia = max(abs(clean_img(:)))   + 1e-12;
    global_scale_bpa = max(abs(clean_img_2(:))) + 1e-12;

    params.global_scale_lia = global_scale_lia;
    params.global_scale_bpa = global_scale_bpa;

    clean_img  = clean_img  / global_scale_lia;
    %target_img = target_img / global_scale_bpa;
else
    global_scale = max(abs(clean_img(:))) + 1e-12;
    params.global_scale = global_scale;

    clean_img  = clean_img  / global_scale;
    %target_img = target_img / global_scale;
end

fprintf('clean_img  abs min/max : %.6e / %.6e\n', ...
    min(abs(clean_img(:))), max(abs(clean_img(:))));
%fprintf('target_img abs min/max : %.6e / %.6e\n', ...
%    min(abs(target_img(:))), max(abs(target_img(:))));

% ---------------------------------------------------------------
% Quick visualization: clean vs shuffled noise-like target
% ---------------------------------------------------------------
%figure;
%subplot(1,2,1);
%imagesc(clean_img); colormap gray; colorbar;
%set(gca, 'YDir', 'normal'); axis image off;
%title(sprintf('Clean Image (%s)', upper(sar_algo)));

%subplot(1,2,2);
%imagesc(target_img); colormap gray; colorbar;
%set(gca, 'YDir', 'normal'); axis image off;
%title('Noise Target Reference');

%% ---------------------------------------------------------------
% Load attack signal pool and build aligned waveform bank D
% ---------------------------------------------------------------
temp_x_aa_1 = load(fullfile(dataDir, "X_aa.mat"));   X_aa_1 = temp_x_aa_1.X_aa;
temp_x_aa_2 = load(fullfile(dataDir, "X_aa_2.mat")); X_aa_2 = temp_x_aa_2.X_aa;

if Nsamp == 512 && M*N > 40000
    X_aa_temp = [X_aa_1; X_aa_2];
    X_aa = [X_aa_temp, X_aa_temp];
elseif Nsamp == 512 && M*N == 40000
    X_aa = [X_aa_1; X_aa_2];
else
    X_aa = X_aa_1;
end

targetK = M*N;
rng(0);
sample_idx = randi(size(X_aa,2), [1, targetK]);
X_a_pool   = X_aa(:, sample_idx);

sel_idx = randi(size(X_a_pool, 2), [1, Np]);
X_a     = X_a_pool(:, sel_idx);

Xspec   = fft(X_a, nFFTtime, 1);
[~, b0] = max(abs(Xspec), [], 1);
f0_col  = (b0 - 1) * FS / nFFTtime;
f_tgt   = (params.k0_range_bin) * FS / nFFTtime;
Delta   = f0_col - f_tgt;

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

        % ---- random complex weights ----
        switch lower(random_weight_mode)
            case 'complex_gaussian'
                A_rand = randn(Np,1) + 1j*randn(Np,1);

            case 'unit_modulus'
                theta = 2*pi*rand(Np,1);
                A_rand = exp(1j*theta);

            otherwise
                error('Unknown random_weight_mode.');
        end

        % ---- scale to target power budget ----
        delta_rand = D .* (A_rand.');
        Pa_rand = norm(delta_rand, 'fro')^2;

        scale_pow = sqrt((target_PaPr * Pr_ref) / (Pa_rand + 1e-12));
        A_rand = A_rand * scale_pow;

        % ---- reconstruct attacked image ----
        [adv_img, PaPr_final, ~] = reconstruct_adv_image_random(X_v, D, A_rand, params);

        % ---- metrics against clean image ----
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
    fprintf('  Success rate (SSIM<=%.2f): %.2f\n\n', success_ssim_thresh, success_rate);

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
    % If none satisfies, choose the strongest tested budget
    chosen_budget_idx = numel(all_budget_results);
    fprintf('No budget met success-rate criterion.\n');
    fprintf('Using strongest tested budget: %.2f dB\n', ...
        all_budget_results(chosen_budget_idx).PaPr_dB);
end

chosen = all_budget_results(chosen_budget_idx);

% Pick the median SSIM_AC trial among successful ones if possible, otherwise best concealment
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
% Final summary for table
% ---------------------------------------------------------------
fprintf('\n============================================================\n');
fprintf('FINAL TABLE NUMBERS for %s on %s\n', upper(sar_algo), rawData(rawData_select));
fprintf('Use budget: %.2f dB\n', chosen.PaPr_dB);
fprintf('PSNR(A,C): %.2f ± %.2f dB\n', chosen.psnr_AC_mean, chosen.psnr_AC_std);
fprintf('SSIM(A,C): %.4f ± %.4f\n', chosen.ssim_AC_mean, chosen.ssim_AC_std);

PaPr_lin_mean = chosen.PaPr_lin;
PaPr_dB_mean  = 10 * log10(PaPr_lin_mean + 1e-12);
fprintf('Pa/Pr     : %.4f (linear mean), %.4f dB\n', PaPr_lin_mean, PaPr_dB_mean);

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
imagesc(clean_img); colormap gray; colorbar;
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
%% ========================================================================

function [adv_img, PaPr_final, PaPr_final_dB] = reconstruct_adv_image_random(X_v, D, A_rand, params)

    A_opt  = A_rand;
    Y_opt  = X_v + D .* (A_opt.');
    Y_cube = reshape(Y_opt, params.Nsamp, params.M, params.N);

    rawDataFFT_att = fft(Y_cube, params.nFFTtime);
    sarData_att    = squeeze(rawDataFFT_att(params.k0_range_bin + 1, :, :));

    for ii = 2:2:size(sarData_att, 1)
        sarData_att(ii, :) = fliplr(sarData_att(ii, :));
    end

    switch upper(params.sar_algo)
        case 'MFA'
            [~, ~, atkImage_abs, ~] = dlMFA(sarData_att, params);
            adv_img = gather(extractdata(atkImage_abs));

        case 'RMA'
            [~, ~, atkImage_abs, ~] = dlRMA(sarData_att, params);
            adv_img = gather(extractdata(atkImage_abs));

        case 'BPA'
            [~, ~, atkImage_abs, ~] = dlBPA(sarData_att, params, params.H_bpa);
            adv_img = gather(extractdata(atkImage_abs));

        case 'LIA'
            [~, ~, atkImage_abs, ~] = dlLIA(sarData_att, params, params.H_bpa);
            adv_img = gather(extractdata(atkImage_abs));

        otherwise
            error('Invalid SAR algorithm selection.');
    end

    if strcmpi(params.sar_algo,'LIA')
        adv_img = adv_img / params.global_scale_lia;
    else
        adv_img = adv_img / params.global_scale;
    end

    delta_opt     = D .* (A_opt.');
    Pa_final      = norm(delta_opt, 'fro')^2;
    Pr_final      = norm(X_v, 'fro')^2 + 1e-12;
    PaPr_final    = Pa_final / Pr_final;
    PaPr_final_dB = 10 * log10(PaPr_final + 1e-12);
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

   
    if strcmpi(params.sar_algo,'LIA')
        atkImage = atkImage / params.global_scale_bpa;   % match target domain (BPA-scale)
    else
        atkImage = atkImage / params.global_scale;
    end
    

    %atkImage = atkImage / params.global_scale;


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

    % NO RMS-normalized magnitude
    img_mag = abs(trueImage_complx);
    % rms_val = sqrt(mean(img_mag(:).^2) + eps);
    % trueImage_abs = dlarray(img_mag ./ rms_val, 'SS');
    trueImage_abs = dlarray(img_mag, 'SS');

    % Spatial ranges (mm)
    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
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
    trueImage_complx = trueImage_cropped; % Complex image output
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
    
    % ---------------------- updated for other sampling variations 
    % old version works only for knife (200x200), updated version will work
    % with other sampling/aperature dimension
    Ny = params.M;      % number of rows in sarData (vertical samples)
    Nx = params.N;      % number of cols in sarData (horizontal samples)
    
    NM = Ny * Nx;
    BA = A * B;
    
    H_val = complex(zeros(NM, BA));
    fprintf('    Building H matrix (%d x %d)...', NM, BA);
    tic;
    
    for i = 1:NM
        % --- MATCHES y = reshape(sarData,[],1) ---
        iy = mod(i-1, Ny);          % 0..Ny-1  (row index)
        ix = (i-1-iy) / Ny;         % 0..Nx-1  (col index)
    
        % sensor coordinates (x uses dx, y uses dy)
        sx_i = (ix + 0.5 - Nx/2) * dxm;
        sy_i = (iy + 0.5 - Ny/2) * dym;
    
        for j = 1:BA
            jy = mod(j-1, B);   jx = (j-1-jy)/B;
            px = wh1(jx+1);
            py = wh2(jy+1);
    
            dist2 = (sx_i - px)^2 + (sy_i - py)^2 + z2;
            H_val(i,j) = exp(cst * sqrt(dist2));
        end
    end
    
    fprintf([' ' num2str(toc, '%.3f') ' sec\n']);
    H = dlarray(H_val);
end

%% %%%%%%%%%%%%%%%%%%%%% MFA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
    % NO RMS-normalize magnitude output
    img_mag = abs(trueImage_cropped);
    %rms_val = sqrt(mean(img_mag(:).^2) + eps);
    %trueImage_abs = dlarray(img_mag ./ rms_val, 'SS');
    trueImage_abs = dlarray(img_mag, 'SS');
    
    % Spatial ranges for plotting
    xRangeT = params.bbox(1) + (0:size(trueImage_abs, 2) - 1) * params.dx;
    yRangeT = params.bbox(3) + (0:size(trueImage_abs, 1) - 1) * params.dy;
end


%% %%%%%%%%%%%%%%%%%%%%% RMA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
    % No RMS-normalize magnitude output
    img_mag = abs(trueImage_cropped);
    %rms_val = sqrt(mean(img_mag(:).^2) + eps);
    %trueImage_abs = img_mag ./ rms_val;
    trueImage_abs = img_mag;
    
    % Set spatial ranges for plotting (consistent with dlMFA)
    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
end



