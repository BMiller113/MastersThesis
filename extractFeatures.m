function [features, validIdx] = extractFeatures(audioFiles, genderType, melMode, cfg)
% extractFeatures
% Mel-feature extraction with optional gender-specific frequency
% ranges and proportional band scaling (prop7k/prop8k). Uses melSpectrogram
% with 'Window' (no deprecated 'WindowLength') (this was a big issue)
% and falls back to STFT+AFB (again, big issue).
%
% Version: 8/15
% Inputs:
%   audioFiles : cellstr / string / char of .wav paths
%   genderType : 'male' | 'female' | 'all'      (optional; default 'all')
%   melMode    : 'default' | 'narrow' | 'wide' | 'prop7k' | 'prop8k' | 'linear'
%                (optional; default 'default')
%
% Outputs:
%   features : [numBands x targetFrames x 1 x N] (single), z-scored globally
%   validIdx : logical [1 x N], false for files that failed robustly


    % ---- Arg defaults ----
    if nargin < 2 || isempty(genderType), genderType = 'all';     end
    if nargin < 3 || isempty(melMode),    melMode    = 'default'; end
    if nargin < 4, cfg = []; end

    % ---- Base settings (overridable via cfg.features) ----
    baseBands     = 40;    % baseline # mel bands (scaled in 'prop*' modes)
    targetFrames  = 32;    % keep time resolution constant
    frameMs       = 25;    % ms window
    hopMs         = 10;    % ms hop (NOT overlap)
    cropMode      = 'center'; % 'center' | 'left' | 'right'

    if ~isempty(cfg) && isstruct(cfg) && isfield(cfg,'features')
        f = cfg.features;
        if isfield(f,'baseBands')    && ~isempty(f.baseBands),    baseBands    = f.baseBands;    end
        if isfield(f,'targetFrames') && ~isempty(f.targetFrames), targetFrames = f.targetFrames; end
        if isfield(f,'frameMs')      && ~isempty(f.frameMs),      frameMs      = f.frameMs;      end
        if isfield(f,'hopMs')        && ~isempty(f.hopMs),        hopMs        = f.hopMs;        end
        if isfield(f,'cropMode')     && ~isempty(f.cropMode),     cropMode     = f.cropMode;     end
    end

    % Convert ms -> seconds
    frameDurSec = frameMs/1000;
    hopDurSec   = hopMs/1000;

    % -------- Normalize input list --------
    if ischar(audioFiles) || isstring(audioFiles)
        audioFiles = cellstr(audioFiles);
    end
    audioFiles = audioFiles(:);
    numFiles   = numel(audioFiles);

    features = [];                 % allocated on first success (locks numBands)
    validIdx = true(1, numFiles);  % per-file success mask

    for i = 1:numFiles
        try
            % Read & precondition
            [audioIn, fs] = tryRead(audioFiles{i});   % robust read
            audioIn = forceMono(audioIn);             % mix to mono if needed
            ma = max(abs(audioIn)); if ma > 0, audioIn = audioIn/ma; end

            % Frame geometry (use HOP to compute OverlapLength)
            frameLength   = max(128, round(frameDurSec * fs));
            hopSamples    = max(1,   round(hopDurSec   * fs));
            overlapLength = max(0,   frameLength - hopSamples);  % MATLAB expects overlap (not hop)

            % Guard: pad to at least one window
            if numel(audioIn) < frameLength
                audioIn(end+1:frameLength,1) = 0;
            end

            % Mel/Filterbank configuration
            [freqRange, numBands, isLinear] = chooseMelConfig(fs, genderType, melMode, baseBands);

            % Allocate on first success (locks band count within this batch)
            ensureAlloc();

            % Front-end: melSpectrogram or custom linear bank
            win = localWindow(frameLength); % hamming with 'periodic' when available
            success = false;

            % Linear filter-bank path (uniform frequency spacing)
            if isLinear
                % STFT/SPECTROGRAM (magnitude)
                if exist('stft','file') == 2
                    S = stft(audioIn, 'Window', win, ...
                        'OverlapLength', overlapLength, ...
                        'FFTLength', frameLength);
                else
                    S = spectrogram(audioIn, win, overlapLength, frameLength, fs);
                end
                % Triangular linear-spaced filterbank
                H = linearTriFilterbank(size(S,1), fs, freqRange(1), freqRange(2), numBands);
                logMel = log10(H * abs(S) + eps);
                success = true;
            end

            % Preferred: melSpectrogram (Audio Toolbox)
            if ~success && exist('melSpectrogram','file') == 2
                % A1: modern call (with FrequencyRange)
                try
                    M = melSpectrogram(audioIn, fs, ...
                        'Window', win, ...
                        'OverlapLength', overlapLength, ...
                        'NumBands', numBands, ...
                        'FrequencyRange', freqRange);
                    logMel = log10(M + eps);
                    success = true;
                catch
                    % A2: older signature (no FrequencyRange)
                    try
                        M = melSpectrogram(audioIn, fs, ...
                            'Window', win, ...
                            'OverlapLength', overlapLength, ...
                            'NumBands', numBands);
                        logMel = log10(M + eps);
                        success = true;
                    catch
                        % continue to fallback
                    end
                end
            end

            % Fallback: STFT/SPECTROGRAM + designAuditoryFilterBank
            if ~success
                if exist('designAuditoryFilterBank','file') ~= 2
                    error('Audio Toolbox function "designAuditoryFilterBank" not found.');
                end

                if exist('stft','file') == 2
                    S = stft(audioIn, 'Window', win, ...
                        'OverlapLength', overlapLength, ...
                        'FFTLength', frameLength);
                else
                    S = spectrogram(audioIn, win, overlapLength, frameLength, fs);
                end

                fb = designAuditoryFilterBank(fs, ...
                    'NumBands', numBands, ...
                    'FFTLength', frameLength, ...
                    'FrequencyRange', freqRange);

                logMel = log10( fb * abs(S) + eps );
                success = true;
            end

            % Last-resort: resample to 16 kHz and retry quickly
            if ~success
                fs2 = 16000;
                audioIn2 = resample(audioIn, fs2, fs);
                frameLength2   = max(128, round(frameDurSec * fs2));
                hopSamples2    = max(1,   round(hopDurSec   * fs2));
                overlapLength2 = max(0,   frameLength2 - hopSamples2);
                win2 = localWindow(frameLength2);
                fr2 = [freqRange(1), min(freqRange(2), fs2/2*0.999)];

                M = melSpectrogram(audioIn2, fs2, ...
                    'Window', win2, ...
                    'OverlapLength', overlapLength2, ...
                    'NumBands', numBands, ...
                    'FrequencyRange', fr2);
                logMel = log10(M + eps);
                success = true;
            end

            % Normalize shape & resize time safely
            % Expect [numBands x frames]. Auto-fix if transposed.
            [r, c] = size(logMel);
            if r ~= numBands && c == numBands
                logMel = logMel.'; [r, c] = size(logMel);
            end
            if r ~= numBands
                error('unexpectedMelShape: got %dx%d, expected %dx*', r, c, numBands);
            end

            % Resize to targetFrames using crop policy
            Z = zeros(numBands, targetFrames, 'like', logMel);
            if c <= targetFrames
                Z(:,1:c) = logMel;   % right-pad
            else
                switch lower(cropMode)
                    case 'left',   startCol = 1;
                    case 'right',  startCol = c - targetFrames + 1;
                    otherwise      % 'center'
                        startCol = floor((c - targetFrames)/2) + 1;
                end
                Z(:,:) = logMel(:, startCol:startCol + targetFrames - 1);
            end
            logMel = Z;

            % Store
            features(:, :, 1, i) = single(logMel);

        catch ME
            % Per-file warning (kept compact)
            warning('extractFeatures:filefail %s | %s', audioFiles{i}, ME.message);
            ensureAlloc();                        % make sure 'features' exists
            features(:, :, 1, i) = 0;             % zero-fill failed item
            validIdx(i) = false;
        end
    end

    % Global Z-score normalization
    mu = mean(features(:));
    sd = std(features(:)) + eps;
    features = (features - mu) / sd;

    % Nested helper
    function ensureAlloc()
        if isempty(features)
            features = zeros(numBands, targetFrames, 1, numFiles, 'single');
        elseif size(features,1) ~= numBands
            error('mixedNumBands: %d (existing) vs %d (this file). Keep melMode constant within a run.', ...
                  size(features,1), numBands);
        end
    end
end

% Helper: pick range & bands 
function [freqRange, numBands, isLinear] = chooseMelConfig(fs, genderType, melMode, baseBands)
    nyq  = fs/2;
    safe = @(x) min(max(0, x), nyq*0.999);

    % Baseline ranges
    switch lower(genderType)
        case 'male',   baseRange = [80, 4000];
        case 'female', baseRange = [150, 6000];
        otherwise,     baseRange = [50, 7000];
    end
    baseRange = [safe(baseRange(1)), safe(baseRange(2))];

    isLinear = false;

    switch lower(melMode)
        case 'default'
            freqRange = baseRange; numBands = baseBands;

        case 'narrow'
            if strcmpi(genderType,'male'),   fr = [80, 4000];
            elseif strcmpi(genderType,'female'), fr = [150, 6000];
            else, fr = baseRange; end
            freqRange = [safe(fr(1)), safe(fr(2))];
            numBands  = baseBands;

        case 'wide'
            if strcmpi(genderType,'male'),   fr = [80, 6000];
            elseif strcmpi(genderType,'female'), fr = [120, 7000];
            else, fr = baseRange; end
            freqRange = [safe(fr(1)), safe(fr(2))];
            numBands  = baseBands;

        case 'prop7k'
            fr = [baseRange(1), safe(7000)];
            freqRange = fr;
            ratio    = diff(fr) / diff(baseRange);
            numBands = roundTo8(baseBands * ratio, 24, 160);

        case 'prop8k'
            fr = [baseRange(1), safe(8000)];
            freqRange = fr;
            ratio    = diff(fr) / diff(baseRange);
            numBands = roundTo8(baseBands * ratio, 24, 200);

        % ---- NEW: Linear frequency filter bank (uniform spacing) ----
        case 'linear'
            isLinear = true;
            if strcmpi(genderType,'female')
                fr = [150, safe(7000)];   % can be [150, 8000] if desired
            else
                fr = baseRange;
            end
            freqRange = [safe(fr(1)), safe(fr(2))];
            numBands  = baseBands;        % keep 40 (or whatever cfg.features.baseBands is)

        otherwise
            error('Unknown mel filter mode: %s', melMode);
    end
end

% Utilities
function n = roundTo8(x, minN, maxN)
    x = max(minN, min(maxN, x));
    n = 8 * max(1, round(x/8));  % prefer multiples of 8 for conv layers
end

function x = forceMono(x)
    if size(x,2) > 1, x = mean(x,2); end
end

function w = localWindow(N)
    % Use 'periodic' when supported; fall back otherwise
    try
        w = hamming(N, 'periodic');
    catch
        w = hamming(N);
    end
end

function [y, fs] = tryRead(path)
    % Try normal read; on failure read 'native' and scale to double
    try
        [y, fs] = audioread(path);
    catch
        [raw, fs] = audioread(path, 'native');
        y = double(raw) / double(intmax(class(raw)));
    end
end

% Linear triangular filterbank helper
function H = linearTriFilterbank(nfftBins, fs, fmin, fmax, numBands)
    % nfftBins: number of positive-freq STFT bins (rows of S)
    % fs: sample rate
    % fmin,fmax: passband (Hz)
    % numBands: number of triangular bands, linearly spaced

    % Frequency axis for the STFT rows (assumes one-sided spectrum)
    freqs = linspace(0, fs/2, nfftBins);
    edges = linspace(max(0,fmin), min(fmax, fs/2*0.999), numBands+2);

    H = zeros(numBands, nfftBins, 'double');
    for b = 1:numBands
        f_left   = edges(b);
        f_center = edges(b+1);
        f_right  = edges(b+2);

        % Rising slope
        idx = freqs >= f_left & freqs <= f_center;
        if any(idx)
            H(b,idx) = (freqs(idx) - f_left) / max(eps, (f_center - f_left));
        end
        % Falling slope
        idx = freqs >= f_center & freqs <= f_right;
        if any(idx)
            H(b,idx) = max(H(b,idx), (f_right - freqs(idx)) / max(eps, (f_right - f_center)));
        end
    end

    % Row-normalize so each band sums to ~1
    s = sum(H,2); s(s==0) = 1;
    H = H ./ s;
end
