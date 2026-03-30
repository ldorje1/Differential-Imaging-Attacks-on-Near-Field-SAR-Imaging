clc; clear; close all;

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

%% ---------------------------------------------------------------
%  User selection: SAR imaging algorithm
% ---------------------------------------------------------------
dataDir = fullfile(pwd, 'data');               % folder holding .mat files
raw_dataDir = fullfile(pwd, 'raw_sar_data');               % folder holding .mat files

%save_final_images = true;   % set to false to skip saving

% ---------------------------------------------------------------
%  Step 2: Load Victim SAR Data and Setup Imaging Parameters
% ---------------------------------------------------------------
%sarRawData = load(fullfile(dataDir, 'rawSAR.mat')).adcDataCube;   % Nsamp x M x N
sar_algo = 'BPA';                              % Options: 'MFA', 'RMA', 'BPA', 'LIA'
target_mode = 'object'; % Options: 'noise' or 'object'

target_raw_select = 1;   % which raw cube to use as OBJECT target (1..10)

rawData_select = 10;   % default selection (change if you want)
rawData = [ ...
    "knife", ...               % 1
    "plier", ...               % 2
    "scissor", ...             % 3
    "screw_driver", ...        % 4
    "sharp_paint_speader", ... % 5
    "dragger",...              % 6
    "wrench", ...              % 7
    "gun", ...                 % 8
    "rifle", ...               % 9
    "butcher_knife" ...              % 10
];

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


    case "butcher_knife",              dx = 1; dy = 2; z0 = 210; FS = 5000e3;
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

        case "butcher_knife",              dx_t = 1; dy_t = 2; z0_tgt = 210; FS_tgt = 5000e3;
        otherwise
            error("Unknown target_raw_select: %s", rawData(target_raw_select));
    end
end

c0 = physconst('lightspeed');
F0 = 77e9;          % start frequency (Hz)
%FS = 5000e3;        % sampling rate (samples/s)
%FS = 9121e3;% Sampling rate (sps)

%Ts = 1 / FS;        % sampling period (s)
K0 = 70.295e12;     % chirp slope (Hz/s)
tI = 4.5225e-10;    % instrument delay (s)

nFFTtime  = 1024;   % range-FFT points
nFFTspace = 1024;   % spatial FFT points (MFA/RMA)

%sarRawData = load(fullfile(dataDir, 'rawSAR.mat')).adcDataCube;   % Nsamp x M x N
sarRawData = load(fullfile(raw_dataDir, rawData(rawData_select) + ".mat")).adcDataCube;   % Nsamp x M x N

if strcmpi(target_mode,'object')
    sarRawData_tgt = load(fullfile(raw_dataDir, rawData(target_raw_select) + ".mat")).adcDataCube;
end

[Nsamp, M, N] = size(sarRawData);

X_v = reshape(sarRawData, Nsamp, M * N);       % Nsamp x Np (vectorize aperture)
Np  = M * N;

% ---------------------------------------------------------------
%  Raw data pre-processing
% ---------------------------------------------------------------
switch upper(sar_algo)
    case 'MFA' % -------------------------------------- MFA
        bbox = [-200 200 -200 200];       % [xmin xmax ymin ymax] in mm

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
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo,'target_mode', target_mode);

        % Generate clean image (reconstruct from correct sarData slice)
        [~, ~, clean_img, ~] = dlMFA(sarData, params);
        clean_img = extractdata(clean_img);
       
        % Generate target image 
        if strcmpi(target_mode,'noise')
            % Target image (structured random): shuffle sarData entries, then re-image
            [rows, cols] = size(sarData); rng(42);                              % fixed seed for reproducibility
            sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
            [~, ~, target_img, ~] = dlMFA(sarData_shuffled, params);
            target_img = extractdata(target_img);

        elseif strcmpi(target_mode,'object')
            % build target params (don’t contaminate victim params)
            params_t = params;
            params_t.dx = dx_t;  params_t.dy = dy_t;  params_t.z0 = z0_tgt;
        
            kbin_t = round(K0 / FS_tgt * (2 * z0_tgt * 1e-3 / c0 + tI) * nFFTtime);
        
            rawDataFFT_t = fft(sarRawData_tgt, nFFTtime);
            sarData_t    = squeeze(rawDataFFT_t(kbin_t + 1, :, :));
        
            for ii = 2:2:size(sarData_t,1)
                sarData_t(ii,:) = fliplr(sarData_t(ii,:));
            end
        
            params_t.k0_range_bin = kbin_t;
        
            [~,~,target_img,~] = dlMFA(sarData_t, params_t);
            target_img = extractdata(target_img);     
  
        else
            error("target_mode must be 'noise' or 'object'.");
        end 
        
        % --- Make target image to match the victim image grid size
        if ~isequal(size(target_img), size(clean_img))
            fprintf('Resizing target_img %s -> %s to match clean_img grid.\n', ...
                mat2str(size(target_img)), mat2str(size(clean_img)));
        
            if exist('imresize','file') == 2
                target_img = imresize(target_img, size(clean_img), 'bilinear');
            else
                % Fallback: interp2 (no Image Processing Toolbox)
                [X, Y]  = meshgrid(linspace(0,1,size(target_img,2)), linspace(0,1,size(target_img,1)));
                [Xq,Yq] = meshgrid(linspace(0,1,size(clean_img,2)),  linspace(0,1,size(clean_img,1)));
                target_img = interp2(X, Y, target_img, Xq, Yq, 'linear', 0);
            end
        end


    case 'RMA' % -------------------------------------- RMA
        % Reorder cube to match your RMA implementation's expected layout
        Echo = permute(sarRawData, [3, 2, 1]);   % [samples, vertical, horizontal] -> [horizontal, vertical, samples]
        bbox = [-200 200 -200 200];

        % RMA uses num_sample FFT points along the sample dimension
        %Nx = 200;
        [Nx, Nz, ~] = size(Echo);
        % different sampling size for knife
        %switch lower(rawData(rawData_select))
        %    case "knife"
        %        Nz = 200;
        %    otherwise
        %        Nz = 125;
        %end

        num_sample = size(Echo, 3);
        nFFTtime   = num_sample;
        rawDataFFT = fft(Echo, nFFTtime, 3);

        % Application-specific selected bin (you treat this as the gate index)
        E = squeeze(sum(sum(abs(rawDataFFT).^2, 1), 2)); 
        [~, k0_range_bin] = max(E);   
   
        % manually set krange,      
        label = lower(rawData(rawData_select));

        switch label
            case {"screw_driver", "gun", "rifle"}
                k0_range_bin = 8;
            case "wrench"
                k0_range_bin = 7;
        end

        %k0_range_bin = 8;

        % Gate the chosen bin and transpose to Nz x Nx style used below
        sarData = squeeze(rawDataFFT(:, :, k0_range_bin)).';
        z0_t = (c0/2) * ( ((k0_range_bin - 1) / (K0*(1/FS)*nFFTtime)) - tI );
       
        % Serpentine correction (flip every other scan row)
        for ii = 2:2:Nz
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % Package parameters used by the imaging operator
        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0_t, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo, 'target_mode', target_mode);

        % Generate clean image (reconstruct from correct sarData slice)
        [~, ~, clean_img, ~] = dlRMA(sarData, params);
        clean_img = extractdata(clean_img);

        % Generate target image 
        if strcmpi(target_mode,'noise')
            % Target image (structured random): shuffle sarData entries, then re-image
            [rows, cols] = size(sarData); rng(42);                              % fixed seed for reproducibility
            sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
            [~, ~, target_img, ~] = dlRMA(sarData_shuffled, params);
            target_img = extractdata(target_img);
        
        elseif strcmpi(target_mode,'object')
            % Build target Echo from target cube
            Echo_t = permute(sarRawData_tgt, [3, 2, 1]);    % [N, M, Nsamp_t]

            num_sample_t = size(Echo_t, 3);
            rawDataFFT_t = fft(Echo_t, num_sample_t, 3);
        
            % Pick target bin by max energy (same as victim RMA)
            E_t = squeeze(sum(sum(abs(rawDataFFT_t).^2, 1), 2));
            [~, kbin_t] = max(E_t);

            % manually set krange, max(E) gives wrong for screw driver
            label_t = lower(rawData(target_raw_select));

            switch label_t
                case {"screw_driver", "gun", "rifle"}
                    kbin_t = 8;
                case "wrench"
                    kbin_t = 7;
            end
            
 
            % Gate target bin
            sarData_t = squeeze(rawDataFFT_t(:, :, kbin_t)).';
            
            [Nx_t, Nz_t, ~] = size(Echo_t);
            % Nz for serpentine (depends on TARGET cube, not victim)
            % Nx_t = 200;
            % switch lower(rawData(target_raw_select))
            %     case "knife"
            %         Nz_t = 200;
            %     otherwise
            %         Nz_t = 125;
            % end
        
            for ii = 2:2:Nz_t
                sarData_t(ii, :) = fliplr(sarData_t(ii, :));
            end
        
            % Target z0 from target bin (seconds -> meters)
            z0_t = (c0/2) * ( ((kbin_t - 1) / (K0*(1/FS_tgt)*num_sample_t)) - tI );
        
            % Build target params (don’t contaminate victim params)
            params_t = params;
            params_t.dx = dx_t;  params_t.dy = dy_t;
            params_t.z0 = z0_t;
            params_t.nFFTtime = num_sample_t;
            params_t.k0_range_bin = kbin_t;
        
            [~, ~, target_img, ~] = dlRMA(sarData_t, params_t);
            target_img = extractdata(target_img);
        else
            error("target_mode must be 'noise' or 'object'.");
        end

        % --- Make target image to match the victim image grid size
        if ~isequal(size(target_img), size(clean_img))
            fprintf('Resizing target_img %s -> %s to match clean_img grid.\n', ...
                mat2str(size(target_img)), mat2str(size(clean_img)));
        
            if exist('imresize','file') == 2
                target_img = imresize(target_img, size(clean_img), 'bilinear');
            else
                % Fallback: interp2 (no Image Processing Toolbox)
                [X, Y]  = meshgrid(linspace(0,1,size(target_img,2)), linspace(0,1,size(target_img,1)));
                [Xq,Yq] = meshgrid(linspace(0,1,size(clean_img,2)),  linspace(0,1,size(clean_img,1)));
                target_img = interp2(X, Y, target_img, Xq, Yq, 'linear', 0);
            end
        end



    case 'BPA' % -------------------------------------- BPA
        
        bbox = [-200 200 -200 200];
       
        % Range FFT along fast-time, then gate a single range bin
        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * (1 / FS) * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        label = lower(rawData(rawData_select));

        switch label
            case {"gun"}
                k0_range_bin = 14;
            case "rifle"
                k0_range_bin = 13;
        end

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
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo, 'target_mode', target_mode);

        % Precompute propagation matrix for BPA
        H_bpa        = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        % Generate clean image (reconstruct from correct sarData slice)
        [~, ~, clean_img, ~] = dlBPA(sarData, params, H_bpa);
        clean_img = extractdata(clean_img);

        if strcmpi(target_mode,'noise')
            % Target image (structured random): shuffle sarData entries, then re-image
            [rows, cols] = size(sarData); rng(42);
            sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
            [~, ~, target_img, ~] = dlBPA(sarData_shuffled, params, H_bpa);
            target_img = extractdata(target_img);

        elseif strcmpi(target_mode,'object')    
            [Nsamp_t, M_t, N_t] = size(sarRawData_tgt);
            params_t = params;
            params_t.M     = M_t;
            params_t.N     = N_t;
            params_t.Nsamp = Nsamp_t;          % not strictly needed for BPA H, but good hygiene
            params_t.dx = dx_t;  params_t.dy = dy_t;  params_t.z0 = z0_tgt;
        
            kbin_t = round(K0 * (1 / FS_tgt) * (2 * z0_tgt * 1e-3 / c0 + tI) * nFFTtime);

            label_t = lower(rawData(target_raw_select));

            switch label_t
                case {"gun"}
                    kbin_t = 14;
                case "rifle"
                    kbin_t = 13;
            end
        
            rawDataFFT_t = fft(sarRawData_tgt, nFFTtime);
            sarData_t    = squeeze(rawDataFFT_t(kbin_t + 1, :, :));
        
            for ii = 2:2:size(sarData_t, 1)
                sarData_t(ii, :) = fliplr(sarData_t(ii, :));
            end
        
            params_t.k0_range_bin = kbin_t;
        
            % Target needs its own H if dx/dy/z0 differ
            H_t        = dlBPA_H_matrix(params_t);
            params_t.H_bpa = H_t;
        
            [~, ~, target_img, ~] = dlBPA(sarData_t, params_t, H_t);
            target_img = extractdata(target_img);
        
        else 
            error("target_mode must be 'noise' or 'object'.");
        end 

        

    case 'LIA' % -------------------------------------- LIA
        bbox = [-200 200 -200 200];
        
        % Range FFT along fast-time, then gate a single range bin
        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * (1/FS) * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);
        label = lower(rawData(rawData_select));

        switch label
            case {"gun"}
                k0_range_bin = 14;
            case "rifle"
                k0_range_bin = 13;
        end

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
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo, 'target_mode', target_mode);

        % Precompute propagation matrix (reused by LIA)
        H_bpa        = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        % Random subset indices for LIA (measurement subsampling)
        NM       = M * N;
        kk       = min(40000, NM);
        rng(1000);
        params.py = sort(randperm(NM, kk));

        % Generate clean image (reconstruct from correct sarData slice)
        [~, ~, clean_img, ~] = dlLIA(sarData, params, H_bpa);
        clean_img = extractdata(clean_img);

        [~, ~, clean_img_2, ~] = dlBPA(sarData, params, H_bpa);
        clean_img_2 = extractdata(clean_img_2);

        % Generate target image 
        if strcmpi(target_mode,'noise')
            % Target image (structured random): shuffle sarData entries, then re-image
            [rows, cols] = size(sarData); rng(42);
            sarData_shuffled = reshape(sarData(randperm(rows*cols)), rows, cols);
            [~, ~, target_img, ~] = dlBPA(sarData_shuffled, params, H_bpa);
            target_img = extractdata(target_img);

        elseif strcmpi(target_mode,'object')
            [Nsamp_t, M_t, N_t] = size(sarRawData_tgt);
            params_t = params;
            params_t.M     = M_t;
            params_t.N     = N_t;
            params_t.Nsamp = Nsamp_t;          % not strictly needed for BPA H, but good hygiene

        
            params_t.dx = dx_t;  params_t.dy = dy_t;  params_t.z0 = z0_tgt;
        
            kbin_t = round(K0 * (1/FS_tgt) * (2 * z0_tgt * 1e-3 / c0 + tI) * nFFTtime);

            label_t = lower(rawData(target_raw_select));

            switch label_t
                case {"gun"}
                    kbin_t = 14;
                case "rifle"
                    kbin_t = 13;
            end

            rawDataFFT_t = fft(sarRawData_tgt, nFFTtime);
            sarData_t    = squeeze(rawDataFFT_t(kbin_t + 1, :, :));
        
            for ii = 2:2:size(sarData_t, 1)
                sarData_t(ii, :) = fliplr(sarData_t(ii, :));
            end
        
            params_t.k0_range_bin = kbin_t;
        
            % Target needs its own H if geometry differs
            H_t          = dlBPA_H_matrix(params_t);
            params_t.H_bpa = H_t;
        
            % Reuse same subsampling rule for target (safe)
            NM_t = size(sarData_t,1) * size(sarData_t,2);
            kk_t = min(40000, NM_t);
            rng(1000);
            params_t.py = sort(randperm(NM_t, kk_t));
        
            %[~, ~, target_img, ~] = dlLIA(sarData_t, params_t, H_t);
            [~, ~, target_img, ~] = dlBPA(sarData_t, params_t, H_t);
            target_img = extractdata(target_img);
        
        else 
            error("target_mode must be 'noise' or 'object'.");
        end

     

    otherwise
        error('Invalid SAR algorithm selection.');

end

% ---------------------------------------------------------------
% Quick visualization: clean vs shuffled-target
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

if strcmpi(sar_algo, 'LIA')
    % LIA scale (for clean visualization / clean reference)
    global_scale_lia = max(abs(clean_img(:)))   + 1e-12;

    % BPA scale (for target + attack objective consistency)
    global_scale_bpa = max(abs(clean_img_2(:))) + 1e-12;

    params.global_scale_lia = global_scale_lia;
    params.global_scale_bpa = global_scale_bpa;

    % Normalize clean by LIA-scale
    clean_img = clean_img / global_scale_lia;

    % Normalize target: object uses LIA-scale, noise(BPA) uses BPA-scale
    %if strcmpi(target_mode,'object')
    target_img = target_img / global_scale_bpa;
    %elseif strcmpi(target_mode,'noise')
    %    target_img = target_img / global_scale_bpa;
    %else
    %    error("target_mode must be 'noise' or 'object'.");
    %end

else
    global_scale = max(abs(clean_img(:))) + 1e-12;
    params.global_scale = global_scale;

    clean_img  = clean_img  / global_scale;
    target_img = target_img / global_scale;
end

% Store dlarray versions for optimization/loss
params.clean_img  = dlarray(clean_img,  "SS");
params.target_img = dlarray(target_img, "SS");



roi_thr = 0.25;   % fraction of peak used to localize object
roi_pad = 6;      % extra pixels around object ROI

if strcmpi(target_mode, 'object')
    roi_box_target = get_roi_box(target_img, roi_thr, roi_pad);
    roi_box_clean  = get_roi_box(clean_img,  roi_thr, roi_pad);

    fprintf('Target ROI [r1 r2 c1 c2] = [%d %d %d %d]\n', roi_box_target);
    fprintf('Clean  ROI [r1 r2 c1 c2] = [%d %d %d %d]\n', roi_box_clean);
else
    roi_box_target = [1 size(target_img,1) 1 size(target_img,2)];
    roi_box_clean  = [1 size(clean_img,1)  1 size(clean_img,2)];
end

function draw_roi_box(box, colorSpec, lineWidth)
% box = [r1 r2 c1 c2]
    r1 = box(1);
    r2 = box(2);
    c1 = box(3);
    c2 = box(4);

    rectangle('Position', [c1, r1, c2-c1+1, r2-r1+1], ...
              'EdgeColor', colorSpec, ...
              'LineWidth', lineWidth);
end

% ---------------------------------------------------------------
% Visualize ROI boxes on clean and target images
% ---------------------------------------------------------------
figure();

subplot(1,2,1);
imagesc(clean_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image;
title('Clean image with ROI');
hold on;
draw_roi_box(roi_box_clean, 'r', 2);
hold off;

subplot(1,2,2);
imagesc(target_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image;
title('Target image with ROI');
hold on;
draw_roi_box(roi_box_target, 'g', 2);
hold off;

%% ---------------------------------------------------------------
%  Step 3: Build Attack Waveforms and Optimize Complex Gains A
% ---------------------------------------------------------------

% Step 1: Load attack signal pool (X_aa)
temp_x_aa_1 = load(fullfile(dataDir, "X_aa.mat")); X_aa_1 = temp_x_aa_1.X_aa;      % loads a .mat file that contains X_aa
temp_x_aa_2 = load(fullfile(dataDir, "X_aa_2.mat")); X_aa_2 = temp_x_aa_2.X_aa;   

% ---------------------------------------------------------------
% 3.1 Select and frequency-align attack waveforms
% ---------------------------------------------------------------
if Nsamp == 512 & M*N > 40000
    X_aa_temp = [X_aa_1; X_aa_2];
    X_aa = [X_aa_temp, X_aa_temp];
elseif Nsamp == 512 & M*N == 40000
    X_aa = [X_aa_1; X_aa_2];
else
    X_aa = X_aa_1;
end

%targetK   = 40000;                                   % number of waveforms sampled from pool
targetK = M*N;
rng(0);                                              % fixed seed for reproducible sampling

%sample_idx = randi(p * p, [1, targetK]);            % (old) alternative indexing if pool size were p*p
sample_idx = randi(size(X_aa,2), [1, targetK]);            % choose targetK random columns from pool [1..16384]
X_a_pool   = X_aa(:, sample_idx);                    % Nsamp x targetK candidate waveforms

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
D      = D .* scale;                                  % RMS-matched injection dictionary

%% ---------------------------------------------------------------
% DIA Optimization
% ---------------------------------------------------------------
% 3.2 Optimization setup (algorithm-specific learning rates)

save_final_images = false;   % set to false to skip saving

switch upper(sar_algo)
    case 'MFA'
        maxIter   = 500;                              % number of gradient iterations
        lr_re     = 1e3;                              % step size for real part of A
        lr_im     = 1e3;                              % step size for imag part of A
        lambda_L2 = 1e-3;                             % L2 regularization weight on A

    case 'RMA'
        maxIter   = 500;
        lr_re     = 1e3;
        lr_im     = 1e3;
        lambda_L2 = 1e-3;


    case 'BPA'
        maxIter   = 300;
        lr_re     = 1e3;
        lr_im     = 1e3;
        lambda_L2 = 1e-3;


    case 'LIA'
        maxIter   = 500;
        lr_re     = 1e3;
        lr_im     = 1e3;
        lambda_L2 = 1e-3;
    otherwise
        error('Invalid SAR algorithm selection.');
end

% -----------------------------------------------------------------
% Constraint options
% -----------------------------------------------------------------
use_pa_pr_projection = false;                          % if true, enforce Pa/Pr <= PaPr_max each iteration
PaPr_max_dB          = -10;                          % 0 dB = equal total power, -10 dB = 10 dB lower, -20 dB = 20 dB lower
PaPr_max             = 10^(PaPr_max_dB/10);          % convert dB power ratio to linear

use_amax_projection  = false;                        % if true, also enforce |A| <= Amax each iteration
Amax                 = 2;                            % per-location amplitude cap (optional)

% Reference signal power (used for Pa/Pr normalization)
Pr_ref = norm(X_v, 'fro')^2 + 1e-12;

% Initialize complex gains A = A_re + j*A_im (Np x 1)
A_re = dlarray(1e-3 * randn(Np, 1, 'double'));       % small random initialization (real)
A_im = dlarray(1e-3 * randn(Np, 1, 'double'));       % small random initialization (imag)

% History buffers for logging/debug
loss_hist     = zeros(maxIter, 1);                   % scalar loss per iteration
meanA_hist    = zeros(maxIter, 1);                   % mean |A| per iteration
maxA_hist     = zeros(maxIter, 1);                   % max  |A| per iteration
PaPr_hist     = zeros(maxIter, 1);                   % Pa/Pr (linear) per iteration
PaPrdB_hist   = zeros(maxIter, 1);                   % Pa/Pr (dB) per iteration

% Console header
fprintf(['\nOptimizing complex gain A for all locations (Np=%d) with SAR algorithm: %s ' ...
         '| lr_re=%.3g, lr_im=%.3g | lambda_L2=%.3g | ' ...
         'Pa/Pr<=%.2f dB (%d) | |A|<=%.3g (%d)\n'], ...
        Np, params.sar_algo, lr_re, lr_im, lambda_L2, ...
        PaPr_max_dB, use_pa_pr_projection, Amax, use_amax_projection);


% ---------------------------------------------------------------
% Main optimization loop (gradient descent on A_re, A_im)
% Constrained: enforce Pa/Pr <= PaPr_max each iteration (projection)
% Stop: early-stop when loss reduction becomes meaningless
%   (1) Diminishing-returns window test (primary)
% ---------------------------------------------------------------

maxIter_cap = maxIter;

% ---- Diminishing-returns early-stop hyperparams (recommended)
check_every   = 25;        % evaluate stop condition every N iters (match your logging)
W             = 50;       % window length in iters
rel_drop_min  = 1e-2;      % 
abs_drop_min  = 0;         % set e.g. 1e-7 if you want an absolute gate too
K_weak        = 3;         % require K consecutive weak windows to stop
weakWinCount  = 0;

% ---- Keep-best tracking (still useful if loss bounces)
bestLoss   = inf;
bestIter   = 0;

bestA_re   = A_re;
bestA_im   = A_im;

iter = 0;

while true
    % cap stop (as you wanted)
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
    A_num = extractdata(A_re) + 1j * extractdata(A_im);   % Np x 1

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
    % Projection 2: enforce Pa/Pr <= PaPr_max (hard constraint)
    % -----------------------------------------------------------
    delta_now = D .* (A_num.');                 % Nsamp x Np
    Pa_now    = norm(delta_now, 'fro')^2;
    PaPr_now  = Pa_now / Pr_ref;

    if use_pa_pr_projection && (PaPr_now > PaPr_max)
        scale_pow = sqrt(PaPr_max / (PaPr_now + 1e-12));
        A_num     = A_num * scale_pow;

        % Recompute after projection (so logs are accurate)
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
    % Update best (simple monotone best; no tol needed)
    % -----------------------------------------------------------
    if lossVal < bestLoss
        bestLoss = lossVal;
        bestIter = iter;
        bestA_re = A_re;
        bestA_im = A_im;

        % reset diminishing-return counter on new best
        %weakWinCount = 0;
    end

    % -----------------------------------------------------------
    % Logging (same style as you had)
    % -----------------------------------------------------------
    if mod(iter, 25) == 0 || iter == 1 || iter == maxIter_cap
        % recompute metrics for current A (post-projection)
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
    % Early stop rule: Diminishing returns over a window
    % -----------------------------------------------------------
    if iter > W && mod(iter, check_every) == 0
        L_old = loss_hist(iter - W);
        L_new = loss_hist(iter);

        rel_drop = (L_old - L_new) / max(abs(L_old), 1e-12);
        abs_drop = (L_old - L_new);

        %isWeak = (rel_drop < rel_drop_min) && (abs_drop < abs_drop_min);
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

% ---- Restore best A (recommended)
A_re = bestA_re;
A_im = bestA_im;

fprintf('-----------------------------\n');
fprintf('Done. Best loss %.4e at iter %d (ran %d iters).\n', bestLoss, bestIter, iter);
fprintf('-----------------------------\n');



% ---------------------------------------------------------------
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
        adv_img = gather(extractdata(atkImage_abs));                 % (kept as-is)

    case 'BPA'
        [~, ~, atkImage_abs, ~] = dlBPA(sarData_att, params, params.H_bpa);
        adv_img = gather(extractdata(atkImage_abs));

    case 'LIA'
        [~, ~, atkImage_abs, ~] = dlLIA(sarData_att, params, params.H_bpa);
        adv_img = gather(extractdata(atkImage_abs));

    otherwise
        error('Invalid SAR algorithm selection for final reconstruction.');
end

% ---------------------------------------------------------------
% ROI setup for object-mode evaluation
% ---------------------------------------------------------------
roi_thr = 0.25;   % fraction of peak used to localize object
roi_pad = 6;      % extra pixels around object ROI

if strcmpi(target_mode, 'object')
    roi_box_target = get_roi_box(target_img, roi_thr, roi_pad);
    roi_box_clean  = get_roi_box(clean_img,  roi_thr, roi_pad);

    fprintf('Target ROI [r1 r2 c1 c2] = [%d %d %d %d]\n', roi_box_target);
    fprintf('Clean  ROI [r1 r2 c1 c2] = [%d %d %d %d]\n', roi_box_clean);
else
    roi_box_target = [1 size(target_img,1) 1 size(target_img,2)];
    roi_box_clean  = [1 size(clean_img,1)  1 size(clean_img,2)];
end

% ---------------------------------------------------------------
% Final metric evaluation (ROI-based for object mode)
% ---------------------------------------------------------------

if strcmpi(params.sar_algo,'LIA')
    adv_img = adv_img / params.global_scale_lia;
else
    adv_img = adv_img / params.global_scale;
end

% --- crops for evaluation ---
adv_t = crop_box(adv_img,    roi_box_target);
tgt_t = crop_box(target_img, roi_box_target);

adv_c = crop_box(adv_img,   roi_box_clean);
cln_c = crop_box(clean_img, roi_box_clean);

% --- A vs T on target ROI ---
AT = compute_metrics_roi(adv_t, tgt_t);

% --- A vs C on clean ROI ---
AC = compute_metrics_roi(adv_c, cln_c);

mse_AT  = AT.mse;
ncc_AT  = AT.ncc;
ssim_AT = AT.ssim;
psnr_AT = AT.psnr;

mse_AC  = AC.mse;
ncc_AC  = AC.ncc;
ssim_AC = AC.ssim;
psnr_AC = AC.psnr;


function box = get_roi_box(img, thr_frac, pad)
% box = [r1 r2 c1 c2]
    A = abs(img);
    A = A / (max(A(:)) + 1e-12);

    rows = find(max(A, [], 2) > thr_frac);
    cols = find(max(A, [], 1) > thr_frac);

    if isempty(rows) || isempty(cols)
        [~, idx] = max(A(:));
        [r, c] = ind2sub(size(A), idx);
        rows = r;
        cols = c;
    end

    r1 = max(rows(1)   - pad, 1);
    r2 = min(rows(end) + pad, size(A,1));
    c1 = max(cols(1)   - pad, 1);
    c2 = min(cols(end) + pad, size(A,2));

    box = [r1 r2 c1 c2];
end

function out = crop_box(img, box)
    out = img(box(1):box(2), box(3):box(4));
end

function S = compute_metrics_roi(x, y)
    S.mse = mean((x(:) - y(:)).^2);

    num = sum(x(:) .* y(:));
    den = sqrt(sum(x(:).^2) * sum(y(:).^2)) + 1e-12;
    S.ncc = num / den;

    dr = max(y(:)) - min(y(:)) + 1e-12;

    if exist('ssim','file') == 2 && min(size(x)) >= 11 && all(size(x) == size(y))
        S.ssim = ssim(x, y, 'DynamicRange', dr);
    else
        S.ssim = NaN;
    end

    if exist('psnr','file') == 2 && all(size(x) == size(y))
        S.psnr = psnr(x, y, dr);
    else
        S.psnr = 10 * log10((dr^2) / (S.mse + 1e-12));
    end
end

% -----------------------------
% Final signal-domain power ratio Pa/Pr
% -----------------------------
delta_opt    = D .* (A_opt.');                        % Nsamp x Np injected signal
Pa_final     = norm(delta_opt, 'fro')^2;             % attack power
Pr_final     = norm(X_v, 'fro')^2 + 1e-12;           % reference power
PaPr_final   = Pa_final / Pr_final;                  % linear ratio
PaPr_final_dB = 10 * log10(PaPr_final + 1e-12);      % dB ratio (optional)

% -----------------------------
% Print metric summary
% -----------------------------
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
hold on;
draw_roi_box(roi_box_clean, 'r', 2);
hold off;

subplot(2,2,2);
imagesc(target_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Target image (desired attacked)');
hold on;
draw_roi_box(roi_box_target, 'g', 2);
hold off;

subplot(2,2,3);
imagesc(adv_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Adversarial image (attacked)');
hold on;
draw_roi_box(roi_box_clean,  'r', 2);   % clean-object ROI
draw_roi_box(roi_box_target, 'g', 2);   % target-object ROI
hold off;

subplot(2,2,4);
imagesc(adv_img - clean_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Diff (A-C)');
hold on;
draw_roi_box(roi_box_clean, 'r', 2);
hold off;
















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



