
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Prepare & Extract Attack Signal Pool (X_aa)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sar_algo = 'RMA';  % Options: 'MFA', 'RMA', 'BPA' <-- SET THE ALGORITHM HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Load IQ data and initialize parameters ---
try
    r0 = load("iqData_noAtk.mat").iqData;   % Nsamp x nRX x nFrame (Clean IQ)
    r1 = load("iqData_Atk.mat").iqData;     % Nsamp x nRX x nFrame (Attacked IQ)
catch ME
    error('Could not load IQ files. Place iqData_noAtk.mat and iqData_Atk.mat in the path.');
end

[Nsamp, nRX, nFrame] = size(r0);
p = nRX * nFrame;
r0_vec = reshape(r0, Nsamp, p); % Nsamp x p (Vectorized measurements)
r1_vec = reshape(r1, Nsamp, p);

% --- Alignment parameters and storage ---
tau_max_guess = 20; % integer delay search window (samples)
tau_search = max(-tau_max_guess, -(Nsamp-1)) : min(tau_max_guess, Nsamp-1);

r_v_est    = zeros(Nsamp, p, 'like', r0_vec);    % Estimated victim (alpha * shifted r0)
r_a_est    = zeros(Nsamp, p, 'like', r0_vec);    % Residual (Extracted attack)
alpha_vec  = complex(zeros(1, p, 'like', r0_vec));% Complex LS scaling factor
tau_samps  = zeros(1, p);                        % Chosen integer delay

fprintf('Running exhaustive tau search + LS scaling on %d channels (p=%d)...\n', p, p);
tic;

% --- Main Alignment Loop (Extracting r_a_est) ---
for col = 1:p
    r0_col = r0_vec(:, col);
    r1_col = r1_vec(:, col);

    % Skip if clean signal is near zero
    if norm(r0_col) < eps
        r_a_est(:, col) = r1_col; % Attack is just the measured signal
        continue;
    end

    best_err = Inf;
    best_alpha = 0;
    best_tau = 0;
    best_r0_shifted = zeros(Nsamp,1,'like',r0_col);

    % Exhaustive integer tau search + LS scaling
    for tau = tau_search
        % Zero-padded non-circular shift
        if tau > 0
            r0_shifted = [zeros(tau,1,'like',r0_col); r0_col(1:end-tau)];
        elseif tau < 0
            t = -tau;
            r0_shifted = [r0_col(t+1:end); zeros(t,1,'like',r0_col)];
        else
            r0_shifted = r0_col;
        end

        % Complex LS scalar: alpha = (r0_shifted' * r1_col) / ||r0_shifted||^2
        denom = (r0_shifted' * r0_shifted);
        alpha = (r0_shifted' * r1_col) / (denom + eps); % add eps for safety
        
        % Compute residual energy
        err = sum(abs(r1_col - alpha * r0_shifted).^2);

        if err < best_err
            best_err = err;
            best_alpha = alpha;
            best_tau = tau;
            best_r0_shifted = r0_shifted;
        end
    end
    
    % Store best results
    alpha_vec(col) = best_alpha;
    tau_samps(col) = best_tau;
    r_v_est(:, col) = best_alpha * best_r0_shifted;
    r_a_est(:, col) = r1_col - r_v_est(:, col); % Extracted attack waveform
end
toc;
fprintf('Finished alignment. mean|alpha| = %.3e\n', mean(abs(alpha_vec)));

% --- Expand aligned residuals into Attack Pool X_aa (Nsamp x p^2) ---
% X_aa(:, k) = r1_vec(:, j) - r_v_est(:, i) for all i,j
vhat = r_v_est;     % vhat(:, i) = alpha_i * shifted(r0(:,i))
p2 = p * p;
X_aa = zeros(Nsamp, p2, 'like', r0_vec);
k = 1;

for i = 1:p
    v_i = vhat(:, i);
    for j = 1:p
        X_aa(:, k) = r1_vec(:, j) - v_i;
        k = k + 1;
    end
end

% --- Column-wise RMS normalization (recommended) ---
colrms = sqrt(mean(abs(X_aa).^2, 1));
X_aa = X_aa ./ (colrms + eps);

stat = @(X)[min(abs(X(:))), max(abs(X(:))), mean(abs(X(:)))];
fprintf('X_aa : size = %d x %d | min=%.3e  max=%.3e  mean=%.3e\n', ...
        size(X_aa,1), size(X_aa,2), stat(X_aa));

%save('X_aa.mat','X_aa');
%--------------------------------------------------------------------------

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: Load Victim SAR Data and Setup Imaging Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Load SAR data and camouflage targets ---
filename_clean = sprintf('trueImage_complex_%s.mat', sar_algo);
filename_attacked = sprintf('desired_attacked_complex_%s.mat', sar_algo);

try
    % Load the raw data (which is static)
    sarRawData = load('rawSAR.mat').adcDataCube; % Nsamp x M x N
    
    % Load the clean complex image (dynamic)
    S_clean = load(filename_clean);
    trueImage_complex = S_clean.trueImage_complex;
    
    % Load the desired attacked complex image (dynamic)
    S_attacked = load(filename_attacked);
    desired_attacked_complex = S_attacked.sar_camouflaged;
    
    % --- Optional: Handle dlarray conversion if necessary ---
    if isa(trueImage_complex, 'dlarray')
        trueImage_complex = extractdata(trueImage_complex);
    end
    if isa(desired_attacked_complex, 'dlarray')
        desired_attacked_complex = extractdata(desired_attacked_complex);
    end

catch ME
    error('Could not load SAR data files for algorithm %s. Check that "rawSAR.mat", "%s", and "%s" are in the path. MATLAB Error: %s', ...
          sar_algo, filename_clean, filename_attacked, ME.message);
end

disp(['Successfully loaded data for SAR algorithm: ', sar_algo]);

[Nsamp_sar, M, N] = size(sarRawData);
assert(Nsamp_sar == Nsamp, 'Sample count mismatch between IQ and SAR data.');

X_v = reshape(sarRawData, Nsamp, M * N); % Nsamp x Np (Vectorized victim data)
Np = M * N; % Total number of aperture locations

% --- SAR Imaging Parameters ---
c0 = physconst('lightspeed');
F0 = 77e9;        % Start frequency (Hz)
FS = 5000e3;      % Sampling rate (sps)
Ts = 1/FS;          % Sampling period
K0 = 70.295e12;   % Slope constant (Hz/sec)
tI = 4.5225e-10;  % Instrument delay (s)

nFFTtime = 1024;  % Range-FFT points
nFFTspace = 1024; % Spatial-FFT points (for MFA)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the processing function handle based on selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch upper(sar_algo)
    case 'MFA'
        dx = 1;           % Horizontal step (mm)
        dy = 1;           % Vertical step (mm)
        bbox = [-500 500 -500 500]; % Bounding box (mm)
        %tI = 4.5225e-10;  % Instrument delay (s)
        z0 = 185;         % Target range (mm)

        % Range focusing bin index (zero-indexed range bin + 1)
        k0_range_bin = round(K0 / FS * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        rawDataFFT = fft(sarRawData, nFFTtime);
        sarData = squeeze(rawDataFFT(k0_range_bin + 1, :, :)); % M x N
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end
        
        % --- Parameter structure for MFA and dlarray targets ---
        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, 'z0', z0, ...
            'dx', dx, 'dy', dy, 'bbox', bbox, 'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
            'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        [~, ~, trueImage_abs, ~] = dlMFA(sarData, params);
        trueImage_abs = extractdata(trueImage_abs);

        plot_sar(trueImage_abs, bbox, dx, dy, 'Clean Reconstructed Image (dlMFA)');

    case 'RMA'
        % [samples, vertical, horizontal]-->[horizontal, vertical, samples]
        Echo = permute(sarRawData, [3,2,1]); % rearrange adcDataCube
        dx = 1;                               % Sampling distance at x (horizontal) axis in mm
        dy = 1;                               % Sampling distance at y (vertical) axis in mm
        bbox = [-500 500 -500 500]; % Bounding box (mm)
         
        Nx = 200;                            % The sampling points in the horizontal direction
        Nz = 200;                            % The sampling points in the vertical direction
        num_sample = size(Echo,3); % 256   
        nFFTtime = num_sample;
        rawDataFFT = fft(Echo,nFFTtime,3); % 407x200x256  % Range FFT
    
        % this original one and this works---------------
        ID_select = 6; % Note: this index for knife image only    
        k0_range_bin = ID_select;
        sarData = squeeze(rawDataFFT(:,:,ID_select)).';                  % Selecting echo data after pulse compression
        z0 = c0/2*(ID_select/(K0*(1/FS)*nFFTtime) - tI);
        for ii = 2:2:Nz
            sarData(ii,:) = fliplr(sarData(ii,:));
        end
    
        % --- Parameter structure for RMA and dlarray targets ---
        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, 'z0', z0, ...
            'dx', dx, 'dy', dy, 'bbox', bbox, 'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
            'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo);

        [~, ~, trueImage_abs, ~] = dlRMA(sarData, params);

        trueImage_abs = extractdata(trueImage_abs);
        
        plot_sar(trueImage_abs, bbox, dx, dy, 'Clean Reconstructed Image (dlRMA)');

    case 'BPA'
        dx = 1;                               % Sampling distance at x (horizontal) axis in mm
        dy = 1;                               % Sampling distance at y (vertical) axis in mm
        bbox = [-500 500 -500 500]; % Bounding box (mm)
        %z0 = 20*1e-3;         % Target range (mm)
        z0   = 190;
        rawDataFFT = fft(sarRawData, nFFTtime);

        k0_range_bin = round(K0*Ts*(2*z0 * 1e-3/c0+tI)*nFFTtime); % corresponing range bin
        sarData = squeeze(rawDataFFT(k0_range_bin+1,:,:));
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        A = 50; % Image Horizontal Pixels
        B = 40; % Image Vertical Pixels

        %H_bpa = dlBPA_H_matrix(params); 

        params = struct('z0', z0,'dx', dx, 'dy', dy, 'bbox', bbox, ...
            'Nsamp', Nsamp, 'nFFTtime',nFFTtime, 'N', N, 'M', M, 'A_bpa', A, ...
            'k0_range_bin', k0_range_bin, 'B_bpa', B, 'F0', F0, ...
            'sar_algo', sar_algo);

        H_bpa = dlBPA_H_matrix(params); 
        params.H_bpa = H_bpa;

        %H_bpa = dlBPA_H_matrix(params); 

        [~, ~, trueImage_abs, ~] = dlBPA(sarData, params, H_bpa);
        trueImage_abs = extractdata(trueImage_abs);

    otherwise
        error('Invalid SAR algorithm selection.');
end

% RMS normalization for target images (required for MSE loss consistency)
true_abs = abs(trueImage_complex);
atk_abs  = abs(desired_attacked_complex);
rms_true = sqrt(mean(true_abs(:).^2) + eps);
rms_atk  = sqrt(mean(atk_abs(:).^2)  + eps);

params.trueImage       = dlarray(true_abs ./ rms_true, "SS");
params.desiredAtkImage = dlarray(atk_abs  ./ rms_atk,  "SS");

% Get the algorithm name (ensuring it's uppercase for consistency)
algo_name = upper(params.sar_algo);

% --- Sanity Check & Visualization ---
figure('Name', ['Target & Reference Images (Algo: ', algo_name, ')'], 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.5]);

if strcmp(upper(sar_algo), 'BPA')
    % Specific, smaller bounding box for BPA
    dx = 1;
    dy = 1;
    bbox = [-500 500 -500 500]; % Bounding box (mm) for BPA
    %plot_sar_bpa(sarImage, bbox, dx, dy, 'true Image (dlBPA)')
    sarImage_clean = extractdata(params.trueImage);
    sarImage_true = extractdata(params.trueImage);
    sarImage_desired = extractdata(params.desiredAtkImage);

    % --- All images use the same brightness or color scale when displayed.
    top_val = max([sarImage_true(:); sarImage_desired(:)]);
    clim    = [0, top_val];   % from 0 to global max
    %-----------------------------------------------------

    subplot(1, 3, 1); 
    plot_sar_bpa(sarImage_clean, bbox, dx, dy, ...
        sprintf('1. Clean Reconstructed Image (%s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

    % 2. LOADED TRUE IMAGE (Reference for loss, RMS-normalized)
    %    This is the clean image loaded from the MAT file.
    subplot(1, 3, 2); 
    plot_sar_bpa(sarImage_true, bbox, dx, dy, ...
        sprintf('2. Loaded True Image (Reference, %s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

    % 3. DESIRED ATTACKED IMAGE (Target for loss, RMS-normalized)
    subplot(1, 3, 3); 
    plot_sar_bpa(sarImage_desired, bbox, dx, dy, ...
        sprintf('3. Desired Attacked Image (Target, %s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

else
    dx = 1;
    dy = 1;
    bbox = [-500 500 -500 500]; % Bounding box (mm) for BPA  

    sarImage_clean = extractdata(params.trueImage);
    sarImage_true = extractdata(params.trueImage);
    sarImage_desired = extractdata(params.desiredAtkImage);

    % --- All images use the same brightness or color scale when displayed.
    top_val = max([sarImage_true(:); sarImage_desired(:)]);
    clim    = [0, top_val];   % from 0 to global max

    subplot(1, 3, 1);

    plot_sar(sarImage_clean, bbox, dx, dy, ...
        sprintf('1. Clean Reconstructed Image (%s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

    % 2. LOADED TRUE IMAGE (Reference for loss, RMS-normalized)
    %    This is the clean image loaded from the MAT file.
    subplot(1, 3, 2); 
    plot_sar(sarImage_true, bbox, dx, dy, ...
        sprintf('2. Loaded True Image (Reference, %s)', algo_name));
    caxis(clim); colorbar; colormap(jet);

    % 3. DESIRED ATTACKED IMAGE (Target for loss, RMS-normalized)
    subplot(1, 3, 3); 
    plot_sar(sarImage_desired, bbox, dx, dy, ...
        sprintf('3. Desired Attacked Image (Target, %s)', algo_name));
    caxis(clim); colorbar; colormap(jet);
end
%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: Simulate Attack, Align Frequency, and Optimize Gain (A)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- 3.1: Select and Frequency Align Attack Waveforms (X_a -> D) ---
targetK = 40000; % Attack pool size to sample from
rng(0);          % Reproducible sampling
sample_idx = randi(p * p, [1, targetK]);
X_a_pool = X_aa(:, sample_idx);

sel_idx = randi(size(X_a_pool, 2), [1, Np]); % One waveform per aperture location
X_a = X_a_pool(:, sel_idx); % Nsamp x Np (Attack waveforms for all locations)

% Per-column frequency shift towards the victim's range bin
Xspec   = fft(X_a, nFFTtime, 1);                 % Nsamp x Np
[~, b0] = max(abs(Xspec), [], 1);                % 1 x Np (peak bins)
f0      = (b0-1) * FS / nFFTtime;               % 1 x Np (peak frequencies)
f_tgt   = (params.k0_range_bin) * FS / nFFTtime;       % Scalar (victim bin frequency)
Delta   = f0 - f_tgt;                            % 1 x Np (required frequency shift)

t = (0:Nsamp-1).' / FS;                          % Nsamp x 1 (time vector in seconds)
P = exp(-1j * 2*pi * (t * Delta));               % Nsamp x Np (Modulator)
D = P .* X_a;                                    % Nsamp x Np (Frequency-shifted base attack)

% Scale D to match X_v RMS column-wise
colrms = @(X) sqrt(mean(abs(X).^2, 1));
scale  = (colrms(X_v)+eps) ./ (colrms(D)+eps);
D = D .* scale;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Main Optimization Loop (Gradient Descent) ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- 3.2: Optimization Setup ---

% --- ALGORITHM-SPECIFIC LEARNING RATES ---
switch upper(sar_algo)
    case 'MFA'
        maxIter= 500; lr_re = 1e2; lr_im = 1e3; lambda_L2 = 1e-4; % L2 regularization on |A| (0 = off) 
    case 'RMA'
        maxIter= 100; lr_re = 1e5; lr_im = 1e5; lambda_L2 = 1e-3; 
    case 'BPA'
        maxIter= 100; lr_re = 1e3; lr_im = 1e3; lambda_L2 = 1e-3; 
    otherwise
        error('Invalid SAR algorithm selection.');
end

use_projection = true; % Optional hard magnitude cap
Amax = 2; 

A_re = dlarray(1e-3 * randn(Np,1,'double')); % Initial random real part
A_im = dlarray(1e-3 * randn(Np,1,'double')); % Initial random imaginary part


% --- Setup history storage
loss_hist = zeros(maxIter,1);
meanA_hist = zeros(maxIter,1);
maxA_hist  = zeros(maxIter,1);

% --- Live plot: Loss, mean|A|, max|A|
figure('Name','Attack Optimization Progress','Color','w');

subplot(3,1,1);
hLoss = semilogy(nan, nan, '-o');  % log scale is usually nice for loss
grid on;
xlabel('Iteration');
ylabel('Loss');
title('Loss vs Iteration');

subplot(3,1,2);
hMeanA = plot(nan, nan, '-o');
grid on;
xlabel('Iteration');
ylabel('mean|A|');
title('Mean |A| vs Iteration');

subplot(3,1,3);
hMaxA = plot(nan, nan, '-o');
grid on;
xlabel('Iteration');
ylabel('max|A|');
title('Max |A| vs Iteration');
%------------------------------------

fprintf('\nOptimizing complex gain A for all locations (Np=%d) with SAR Imaging Algorithm: %s\n ...\n', Np, params.sar_algo); 
for iter = 1:maxIter
    % Loss and gradients wrt A_re, A_im
    [loss, gRe, gIm] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);

    % Convert loss dlarray -> double safely
    lossVal = double(gather(extractdata(loss)));
    loss_hist(iter) = lossVal;

    % Gradient steps
    A_re = A_re - lr_re * gRe;
    A_im = A_im - lr_im * gIm;
    
    % Optional projection
    if use_projection
        A_num = extractdata(A_re) + 1j * extractdata(A_im);
        mags = abs(A_num);
        over = mags > Amax;
        if any(over)
            scale_proj = ones(size(mags));
            scale_proj(over) = Amax ./ mags(over);
            A_num = A_num .* scale_proj;
            A_re  = dlarray(real(A_num));
            A_im  = dlarray(imag(A_num));
        end
    end
    
    % Record A stats
    A_now        = extractdata(A_re) + 1j*extractdata(A_im);
    meanA_hist(iter) = mean(abs(A_now));
    maxA_hist(iter)  = max(abs(A_now));

    % --- Live plot update ---
    iters = 1:iter;

    % Update loss curve
    set(hLoss, 'XData', iters, 'YData', loss_hist(1:iter));

    % Update mean|A|
    set(hMeanA, 'XData', iters, 'YData', meanA_hist(1:iter));

    % Update max|A|
    set(hMaxA, 'XData', iters, 'YData', maxA_hist(1:iter));

    drawnow limitrate;   % 'limitrate' avoids overloading the UI

    % Text logging (optional)
    if mod(iter,10)==0 || iter==1 || iter==maxIter
        fprintf('Iter %3d/%3d | Loss=%.6e | mean|A|=%.3e, max|A|=%.3e\n', ...
            iter, maxIter, lossVal, meanA_hist(iter), maxA_hist(iter));
    end
end
fprintf('-----------------------------\n');


% fprintf('\nOptimizing complex gain A for all locations (Np=%d) with SAR Imaging Algorithm: %s\n ...\n', Np, params.sar_algo);
% for iter = 1:maxIter
%     % Loss and gradients wrt A_re, A_im
%     [loss, gRe, gIm] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);
% 
% 
%     % Convert loss dlarray -> double safely
%     lossVal = double(gather(extractdata(loss)));
%     loss_hist(iter) = lossVal;
% 
%     % Gradient steps
%     A_re = A_re - lr_re * gRe;
%     A_im = A_im - lr_im * gIm;
% 
%     % Optional projection
%     if use_projection
%         A_num = extractdata(A_re) + 1j * extractdata(A_im);
%         mags = abs(A_num);
%         over = mags > Amax;
%         if any(over)
%             scale_proj = ones(size(mags));
%             scale_proj(over) = Amax ./ mags(over);
%             A_num = A_num .* scale_proj;
%             A_re = dlarray(real(A_num));
%             A_im = dlarray(imag(A_num));
%         end
%     end
% 
%     % Record A stats
%     A_now = extractdata(A_re) + 1j*extractdata(A_im);
%     meanA_hist(iter) = mean(abs(A_now));
%     maxA_hist(iter)  = max(abs(A_now));
% 
%     % Logging
%     if mod(iter,10)==0 || iter==1 || iter==maxIter
%         A_now = extractdata(A_re) + 1j*extractdata(A_im);
%         fprintf('Iter %3d/%3d | Loss=%.6e | mean|A|=%.3e, max|A|=%.3e\n', ...
%             iter, maxIter, extractdata(loss), mean(abs(A_now)), max(abs(A_now)));
%     end
% end
% fprintf('-----------------------------\n');

%%
% --- 3.3: Reconstruct Attacked Image with Optimal A ---
A_opt = extractdata(A_re) + 1j * extractdata(A_im); % Np x 1
Y_opt = X_v + D .* (A_opt.');                       % Nsamp x Np (Final attacked raw data)
Y_cube = reshape(Y_opt, Nsamp, M, N);
rawDataFFT_att = fft(Y_cube, nFFTtime);
sarData_att = squeeze(rawDataFFT_att(k0_range_bin + 1, :, :)); % M x N (Range-compressed slice)

% Correction
for ii = 2:2:size(sarData_att, 1)
    sarData_att(ii, :) = fliplr(sarData_att(ii, :));
end

% --- FIX: DYNAMIC RECONSTRUCTION BASED ON ALGO ---
switch upper(params.sar_algo)
    case 'MFA'
        [~, ~, atkImage_abs, ~] = dlMFA(sarData_att, params);
        I_att = gather(extractdata(atkImage_abs));
        % Display Attacked Image
        figure('Name','Attacked Image (A_{opt})','Color','w');
        plot_sar(I_att, bbox, dx, dy, 'Attacked Image (Reconstructed using A_{opt})');
        caxis(clim); colorbar; colormap(jet);
    case 'RMA'
        [~, ~, atkImage_abs, ~] = dlRMA(sarData_att, params);
        I_att = gather(extractdata(atkImage_abs));

        % Display Attacked Image
        figure('Name','Attacked Image (A_{opt})','Color','w');
        plot_sar(I_att, bbox, dx, dy, 'Attacked Image (Reconstructed using A_{opt})');
        caxis(clim); colorbar; colormap(jet);
    case 'BPA'
        % For BPA, must pass the H-matrix which is stored in params.H_bpa
        [~, ~, atkImage_abs, ~] = dlBPA(sarData_att, params, params.H_bpa);
        I_att = gather(extractdata(atkImage_abs));
        plot_sar_bpa(I_att, params.bbox, params.dx, params.dy, ...
        sprintf('Attacked Image (Reconstructed using A_{opt})'));
        caxis(clim); colorbar; colormap(jet);
    otherwise
        error('Invalid SAR algorithm selection for final reconstruction.');
end

%%
% --- Convert to dB scale ---
mag0 = true_abs;
mag0_db = 20*log10(mag0 + eps);

mag1 = I_att;
mag1_db = 20*log10(mag1 + eps);

% --- Create comparison figure ---
figure('Color','w','Name','Before vs After Attack','Units','normalized','Position',[0.1 0.1 0.8 0.4]);

subplot(1,2,1);
imagesc(mag0_db);
axis image off; colormap(jet); colorbar;
set(gca,'YDir','normal'); caxis(clim);
title('Before (dB)');

subplot(1,2,2);
imagesc(mag1_db);
axis image off; colormap(jet); colorbar;
set(gca,'YDir','normal'); caxis(clim);
title('After Attack (dB)');

sgtitle('SAR Image Comparison: Before vs After Attack');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4: Evaluation Quantitative Attack Assessment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_ref = extractdata(params.desiredAtkImage); % Target image (RMS-normalized)
fprintf('\n--- Spoofing Attack Performance ---\n');

% 1. Mean Squared Error (MSE) - (Lower is better)
mse_val = mean((I_att(:) - I_ref(:)).^2);
fprintf('Mean Squared Error (MSE): %.6e\n', mse_val);

% 2. Root Mean Squared Error (RMSE) - (Lower is better)
rmse_val = sqrt(mse_val);
fprintf('Root Mean Squared Error (RMSE): %.6e\n', rmse_val);

% 3. Normalized Cross-Correlation (NCC) - (Target: 1.0, Higher is better)
I_ref_vec = I_ref(:);
I_att_vec = I_att(:);
numerator = sum(I_ref_vec .* I_att_vec);
denominator = sqrt(sum(I_ref_vec.^2) * sum(I_att_vec.^2));
ncc_val = numerator / (denominator + eps); % Add epsilon for stability
fprintf('Normalized Cross-Correlation (NCC): %.4f\n', ncc_val);

% 4. Structural Similarity Index (SSIM) - (Requires Image Processing Toolbox, Target: 1.0)
if license('test', 'Image_Toolbox')
    ssim_val = ssim(I_att, I_ref, 'DynamicRange', max(I_ref(:)));
    fprintf('Structural Similarity Index (SSIM): %.4f\n', ssim_val);
else
    fprintf('Structural Similarity Index (SSIM): Skipping (Image Processing Toolbox required).\n');
end

% 5. Peak Signal-to-Noise Ratio (PSNR) - (Requires Image Processing Toolbox, Higher is better)
if license('test', 'Image_Toolbox')
    % Max value is usually taken as max(I_ref(:)) or 1 since the data is RMS-normalized.
    max_val = max(I_ref(:)); 
    psnr_val = psnr(I_att, I_ref, max_val);
    fprintf('Peak Signal-to-Noise Ratio (PSNR): %.2f dB\n', psnr_val);
else
    fprintf('Peak Signal-to-Noise Ratio (PSNR): Skipping (Image Processing Toolbox required).\n');
end

%--------------------------------------------------------------------------

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 5: Visual Comparison Panel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name','Attack Evaluation: Image Comparison','Units','normalized','Position',[0.05 0.1 0.9 0.45]);

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
imagesc(abs(I_att - I_ref));
title('|I_{att} - I_{ref}| (Error Map)');
axis image off; colormap hot; colorbar;
sgtitle('Spoofing Attack Visual Assessment');


% --- Scatter Plot: Pixel Intensity Correlation ---
figure('Name','Pixel Correlation','Units','normalized','Position',[0.3 0.3 0.4 0.4]);
scatter(I_ref(:), I_att(:), 5, '.'); hold on;

% Force axes to start at 0, add 5% padding on the upper bound
maxVal = max([max(I_ref(:)), max(I_att(:))]);
pad = 0.05 * maxVal;
plot([0, maxVal+pad], [0, maxVal+pad], 'r--', 'LineWidth', 1.2);

xlabel('Target Intensity I_{ref}');
ylabel('Attacked Intensity I_{att}');
axis equal; grid on;
xlim([0, maxVal+pad]);
ylim([0, maxVal+pad]);
title(sprintf('Pixel Correlation (NCC=%.3f, SSIM=%.3f)', ncc_val, ssim_val));


% Histogram of Pixel-wise Errors
figure('Name','Error Histogram','Units','normalized','Position',[0.3 0.3 0.4 0.4]);
histogram(I_att(:) - I_ref(:), 100);
xlabel('Pixel Error (I_{att} - I_{ref})');
ylabel('Count');
title('Error Distribution');
grid on;

% Log-Spectrum Comparison
figure('Name','Log-Spectrum Comparison','Units','normalized','Position',[0.05 0.1 0.9 0.4]);
subplot(1,3,1);
imagesc(log10(abs(fftshift(fft2(I_ref)))+1e-12)); axis image off;
title('Target Spectrum'); colormap jet; colorbar;

subplot(1,3,2);
imagesc(log10(abs(fftshift(fft2(I_att)))+1e-12)); axis image off;
title('Attacked Spectrum'); colormap jet; colorbar;

subplot(1,3,3);
imagesc(log10(abs(fftshift(fft2(I_att)))+1e-12) - log10(abs(fftshift(fft2(I_ref)))+1e-12));
axis image off; colormap jet; colorbar;
title('Spectral Difference (dB)');
sgtitle('Frequency-Domain Similarity');

%  Overlay Error on Target
err_map = abs(I_att - I_ref);
overlay = mat2gray(I_ref);
overlay_rgb = ind2rgb(gray2ind(overlay,256), jet(256));
err_overlay = imoverlay(overlay_rgb, err_map > 0.1*max(err_map(:)), [1 0 0]); % threshold
figure; imshow(err_overlay); title('Error Overlay (Red = High Discrepancy)');

% ----------------------------------------------------------------------
% VISUALIZATION: 3D Surface Plots for Clean | Desired | Attacked | Error
% ----------------------------------------------------------------------

% --- 1. Prepare Data and Spatial Coordinates ---
% Create the pixel grid for plotting
[y_idx, x_idx] = size(I_ref);
% Create physical coordinates for X and Y axes (using original bbox/dx/dy)
xv = bbox(1) + (0:x_idx-1) * dx;
yv = bbox(3) + (0:y_idx-1) * dy;
[X, Y] = meshgrid(xv, yv);

error_surface = abs(I_att - I_ref);

% Set a consistent Z-limit for the first three magnitude plots
ZLim_mag = [0, max([max(trueImage_abs(:)), max(I_ref(:)), max(I_att(:))])];
ZLim_err = [0, max(error_surface(:))];

% --- 2. Create Figure and 2x2 Subplots ---
figure('Name','3D Camouflage Attack Surface Analysis','Units','normalized','Position',[0.1 0.1 0.8 0.8]);

% Define a common viewing angle
view_angle = [30, 45]; % [Azimuth, Elevation]

% --- Subplot 1: Clean Reconstructed Image (3D) ---
subplot(2, 2, 1);
surf(X, Y, trueImage_abs, 'EdgeColor', 'none', 'FaceColor', 'interp');
caxis(ZLim_mag); zlim(ZLim_mag);
colormap(gca, 'jet'); colorbar;
title('1. Clean Image (I_{clean})');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Magnitude');
view(view_angle);

% --- Subplot 2: Desired Camouflage Target (3D) ---
subplot(2, 2, 2);
surf(X, Y, I_ref, 'EdgeColor', 'none', 'FaceColor', 'interp');
caxis(ZLim_mag); zlim(ZLim_mag);
colormap(gca, 'jet'); colorbar;
title('2. Desired Attacked Target (I_{ref})');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Magnitude');
view(view_angle);

% --- Subplot 3: Final Attacked Image (3D) ---
subplot(2, 2, 3);
surf(X, Y, I_att, 'EdgeColor', 'none', 'FaceColor', 'interp');
caxis(ZLim_mag); zlim(ZLim_mag);
colormap(gca, 'jet'); colorbar;
title('3. Attacked Result (I_{att})');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Magnitude');
view(view_angle);

% --- Subplot 4: 3D Error Surface (|I_att - I_ref|) ---
subplot(2, 2, 4);
surf(X, Y, error_surface, 'EdgeColor', 'none', 'FaceColor', 'interp');
caxis(ZLim_err); zlim(ZLim_err);
colormap(gca, 'hot'); colorbar; % Use 'hot' colormap to highlight high errors
title('4. |I_{att} - I_{ref}| (3D Error Surface)');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Error Magnitude');
view(view_angle);

% Link the view of the surface plots for consistent rotation (optional but useful)
linkprop([subplot(2,2,1), subplot(2,2,2), subplot(2,2,3), subplot(2,2,4)],'CameraPosition','CameraTarget','CameraUpVector');
%%
% test for shift
C = normxcorr2(I_ref, I_att);
[peak, idx] = max(C(:));
[ypeak, xpeak] = ind2sub(size(C), idx);
shift_y = ypeak - size(I_ref,1);
shift_x = xpeak - size(I_ref,2);
fprintf('normxcorr2 peak=%.3f at shift (%d, %d)\n', peak, shift_y, shift_x);



%% Helper Functions (SAR Processing and Visualization)

% function sarImage = visualize_image(sarRawData, params)
% % Helper to run the standard processing chain for visualization
%     rawDataFFT = fft(sarRawData, params.nFFTtime);
%     sarData = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :)); % M x N
%     for ii = 2:2:size(sarData, 1)
%         sarData(ii, :) = fliplr(sarData(ii, :));
%     end
%     [~, ~, trueImage_abs, ~] = dlMFA(sarData, params);
%     sarImage = extractdata(trueImage_abs);
% end

function plot_sar(sarImage, bbox, dx, dy, plotTitle)
% Helper for standard mesh plot
    xv = bbox(1) + (0:size(sarImage, 2) - 1) * dx;
    yv = bbox(3) + (0:size(sarImage, 1) - 1) * dy;
    mesh(xv, yv, sarImage, 'FaceColor', 'interp', 'LineStyle', 'none');
    view(2); colormap('jet');
    axis equal tight; xlabel('Horizontal (mm)'); ylabel('Vertical (mm)');
    title(plotTitle); xlim([bbox(1) bbox(2)]); ylim([bbox(3) bbox(4)]);
end



function [loss, gradRe, gradIm] = loss_and_grad(X_v, D, A_re, A_im, params, lambda_L2)
% LOSS_AND_GRAD: Computes MSE loss and gradients for optimization
% Supports: params.sar_algo = 'MFA' | 'RMA' | 'BPA'

    % ------------- 0) prep -------------
    if ~isa(X_v, 'dlarray'), X_v = dlarray(X_v); end
    if ~isa(D,   'dlarray'), D   = dlarray(D);   end
    if nargin < 6 || isempty(lambda_L2), lambda_L2 = 0; end

    A = A_re + 1j * A_im;                 % (Np x 1)
    Y = X_v + D .* A.';                   % (Nsamp x Np)

    % reshape back to cube: (Nsamp x M x N)
    Y_cube = reshape(Y, params.Nsamp, params.M, params.N);

    % ------------- 1) pick algorithm -------------
    algo = upper(params.sar_algo);  % make sure you set this in the main script
    switch algo

        case 'MFA'
            % range-FFT along fast time
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :)); % M x N

            % serpentine
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end

            % reconstruct
            [~, ~, atkImage, ~] = dlMFA(sarData, params);

        case 'RMA'
            % range-FFT along fast time
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :)); % M x N

            % serpentine
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end

            % RMA expects MxN just like MFA
            [~, ~, atkImage, ~] = dlRMA(sarData, params);

        case 'BPA'
            % For BPA we don't re-pick a single range bin — we assume Y is already
            % at the chosen range bin (like what you did before calling dlBPA)
            %
            % So: Y_cube: (Nsamp x M x N), but BPA is built on MxN 2D data.
            % Take the range-compressed slice the same way you did in the main code:
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :)); % M x N

            % serpentine
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end

            % Need H_bpa in params
            if ~isfield(params, 'H_bpa')
                error('params.H_bpa is required for BPA in loss_and_grad.');
            end

            [~, ~, atkImage, ~] = dlBPA(sarData, params, params.H_bpa);

        otherwise
            error('Unknown params.sar_algo = %s', params.sar_algo);
    end
    % ------------- 2) loss -------------
    loss_im = mean((atkImage - params.desiredAtkImage).^2, 'all');

    % L2 on A
    reg = lambda_L2 * mean(abs(A).^2, 'all');

    loss = loss_im + reg;

    % ------------- 3) grads -------------
    [gradRe, gradIm] = dlgradient(loss, A_re, A_im);
end

% function [loss, gradRe, gradIm] = loss_and_grad(X_v, D, A_re, A_im, params, lambda_L2)
% % LOSS_AND_GRAD: Computes MSE loss and gradients for optimization.
%     % Ensure inputs are dlarray
%     if ~isa(X_v, 'dlarray'), X_v = dlarray(X_v); end
%     if ~isa(D,   'dlarray'), D   = dlarray(D);   end
% 
%     A = A_re + 1j * A_im;              % Np x 1 complex dlarray
%     Y = X_v + D .* A.';                % Nsamp x Np (Attacked raw data)
% 
%     Y_cube = reshape(Y, params.Nsamp, params.M, params.N);
%     rawDataFFT = fft(Y_cube, params.nFFTtime);
%     sarData = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :)); % M x N
% 
%     % Serpentine correction
%     for ii = 2:2:size(sarData, 1)
%         sarData(ii, :) = fliplr(sarData(ii, :));
%     end
% 
%     [~, ~, atkImage, ~] = dlMFA(sarData, params);
% 
%     % Compute MSE loss (Camouflage objective)
%     loss_im = mean((atkImage - params.desiredAtkImage) .^ 2, 'all');
% 
%     % L2 regularizer on A
%     if nargin < 6 || isempty(lambda_L2), lambda_L2 = 0; end
%     reg = lambda_L2 * mean(abs(A).^2, 'all');
% 
%     loss = loss_im + reg;
% 
%     % Compute gradients
%     [gradRe, gradIm] = dlgradient(loss, A_re, A_im);
% end

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
    trueImage_complx = trueImage_cropped;
    
    % RMS-normalize magnitude output
    img_mag = abs(trueImage_cropped);
    rms_val = sqrt(mean(img_mag(:).^2) + eps);
    trueImage_abs = img_mag ./ rms_val;
    
    % Set spatial ranges for plotting (consistent with dlMFA)
    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
end

% function plot_sar_bpa(I_plot, bbox, dx, dy, plotTitle)
%     % I_plot: B x A image (vertical x horizontal)
%     % bbox : [xmin xmax ymin ymax] in mm
%     % dx,dy: mm per pixel
% 
%     % x / y axes in mm
%     xv = bbox(1) + (0:size(I_plot, 2)-1) * dx;
%     yv = bbox(3) + (0:size(I_plot, 1)-1) * dy;
% 
%     mesh(xv, yv, I_plot, 'FaceColor', 'interp', 'LineStyle', 'none');
%     view(2);
%     colormap('jet');
%     axis equal tight;
%     xlabel('Horizontal (mm)');
%     ylabel('Vertical (mm)');
%     title(plotTitle);
% 
%     xlim([bbox(1) bbox(2)]);
%     ylim([bbox(3) bbox(4)]);
% end

function plot_sar_bpa(I_plot, bbox, dx, dy, plotTitle)
    % PLOT_SAR_BPA: Plots the image, using linspace to define axes based on bbox.
    % This structure is consistent with how BPA often defines its grid.
    %
    % I_plot: B x A image (vertical x horizontal)
    % bbox : [xmin xmax ymin ymax] in mm
    % dx,dy: mm per pixel - NOTE: dx/dy are primarily used for coordinate scaling
    %                             in this linspace approach, they are implicitly
    %                             defined by bbox and A/B.

    % Get image dimensions B x A (Vertical x Horizontal pixels)
    [B, A] = size(I_plot);

    % --- 1. Define Image Coordinates using LINSPACE ---
    % Since bbox is in mm, wh1/wh2 will also be in mm.
    % We use A and B (horizontal and vertical size) to define the spacing.
    % The coordinates are automatically in mm.
    xv = linspace(bbox(1), bbox(2), A); % Horizontal axes (mm)
    yv = linspace(bbox(3), bbox(4), B); % Vertical axes (mm)

    % --- 2. Execute Plot (similar to your desired format) ---
    %figure;
    mesh(xv, yv, I_plot, 'FaceColor', 'interp', 'LineStyle', 'none');
    
    view(2);
    colormap('jet');
    axis equal tight; % Use 'tight' to fit data limits
    
    xlabel('Horizontal (mm)');
    ylabel('Vertical (mm)');
    title(plotTitle);

    xlim([bbox(1) bbox(2)]);
    ylim([bbox(3) bbox(4)]);
end
