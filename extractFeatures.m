function [features, validIdx] = extractFeatures(audioFiles, genderType, melMode)
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
%   melMode    : 'default' | 'narrow' | 'wide' | 'prop7k' | 'prop8k'
%                (optional; default 'default')
%
% Outputs:
%   features : [numBands x targetFrames x 1 x N] (single), z-scored globally
%   validIdx : logical [1 x N], false for files that failed robustly

    % Defaults (for 1 arg saftey)
    if nargin < 2 || isempty(genderType), genderType = 'all';     end
    if nargin < 3 || isempty(melMode),    melMode    = 'default'; end

    % Base settings
    baseBands       = 40;    % baseline # mel bands (scaled in 'prop*' modes)
    targetFrames    = 32;    % keep time resolution constant
    frameDuration   = 0.025; % 25 ms
    overlapDuration = 0.010; % 10 ms

    % Normalize input list
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

            frameLength   = max(128, round(frameDuration   * fs));
            overlapLength = round(overlapDuration * fs);

            % Guard: pad to at least one window
            if numel(audioIn) < frameLength
                audioIn(end+1:frameLength,1) = 0;
            end

            % Mel config
            [freqRange, numBands] = chooseMelConfig(fs, genderType, melMode, baseBands);

            % Allocate on first success
            ensureAlloc();

            %  MelSpectrogram
            win = localWindow(frameLength); % hamming with 'periodic' when available
            success = false;

            if exist('melSpectrogram','file') == 2
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
                frameLength2   = max(128, round(frameDuration * fs2));
                overlapLength2 = round(overlapDuration * fs2);
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

            % Normalize shape
            % Expect [numBands x frames]. Fix if transposed
            [r, c] = size(logMel);
            if r ~= numBands && c == numBands
                logMel = logMel.'; [r, c] = size(logMel);
            end
            if r ~= numBands
                error('unexpectedMelShape: got %dx%d, expected %dx*', r, c, numBands);
            end

            % Resize without [] concat (shape-safe)
            Z = zeros(numBands, targetFrames, 'like', logMel);
            if c == 0
                % keep zeros
            elseif c <= targetFrames
                Z(:,1:c) = logMel;
            else
                startCol = floor((c - targetFrames)/2) + 1;
                Z(:,:) = logMel(:, startCol:startCol + targetFrames - 1);
            end
            logMel = Z;

            % Storage
            features(:, :, 1, i) = single(logMel);

        catch ME
            % Per-file warning
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
function [freqRange, numBands] = chooseMelConfig(fs, genderType, melMode, baseBands)
    nyq  = fs/2;
    safe = @(x) min(max(0, x), nyq*0.999);

    % Baseline ranges
    switch lower(genderType)
        case 'male',   baseRange = [80, 4000];
        case 'female', baseRange = [150, 6000];
        otherwise,     baseRange = [50, 7000];
    end
    baseRange = [safe(baseRange(1)), safe(baseRange(2))];
    baseWidth = diff(baseRange);

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
            ratio    = diff(fr) / baseWidth;
            numBands = roundTo8(baseBands * ratio, 24, 160);

        case 'prop8k'
            fr = [baseRange(1), safe(8000)];
            freqRange = fr;
            ratio    = diff(fr) / baseWidth;
            numBands = roundTo8(baseBands * ratio, 24, 200);

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
