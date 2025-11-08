function S = slidingWindowScores(net, wavPath, cfg)
% slidingWindowScores  Run your utterance CNN on sliding windows of a stream.
% Returns:
%   S.sr          : sample rate used (16 kHz)
%   S.t_center    : [Nwin x 1] time (s) at center of each window
%   S.winSpanSec  : window duration in seconds
%   S.scores      : [Nwin x C] softmax scores (same class order as net)
%   S.classNames  : {1xC} class names (cellstr)
%
% Implementation matches your training front-end:
% - Uses cfg.features.frameMs/hopMs/baseBands/targetFrames
% - Default windowMs ≈ frameMs + (targetFrames-1)*hopMs  (i.e., the same
%   receptive field used at train time), but can be overridden via cfg.streaming.windowMs
% - Hop between windows via cfg.streaming.hopWinMs (default 100 ms)

% ---------- Network classes/order ----------
classNames = getNetClasses(net);

% ---------- Audio load ----------
sr = 16000;
[x, fs] = audioread(wavPath);
if size(x,2) > 1, x = mean(x,2); end
if fs ~= sr, x = resample(x, sr, fs); end
x = single(max(-1,min(1,x)));

L = numel(x);

% ---------- Front-end geometry from cfg.features ----------
frameMs     = getf(cfg,'features','frameMs', 25);
hopMs       = getf(cfg,'features','hopMs',   10);
baseBands   = getf(cfg,'features','baseBands',   40);
targetFrames= getf(cfg,'features','targetFrames',32);

frameLen    = max(128, round(sr * frameMs/1000));
hopSamp     = max(1,   round(sr * hopMs /1000));
overlapLen  = max(0, frameLen - hopSamp);

% Default window length to match training receptive field; allow override
defaultWinMs = frameMs + (targetFrames-1)*hopMs;
windowMs  = getf(cfg,'streaming','windowMs', defaultWinMs);
hopWinMs  = getf(cfg,'streaming','hopWinMs', 100);   % 10 Hz default
winLen    = round(sr * windowMs/1000);
hopWin    = max(1, round(sr * hopWinMs/1000));

% time centers
tCenters = ((0:hopWin:(L-winLen)) + winLen/2) / sr;
Nwin = numel(tCenters);
if Nwin==0
    S = struct('sr',sr,'t_center',[], 'winSpanSec',windowMs/1000, ...
               'scores',[], 'classNames',{classNames});
    return;
end

% ---------- Feature & batch assembly ----------
% Frequency range like your 'default' ALL setting
fr = chooseDefaultFreqRange(sr);
win = localHamming(frameLen);
X = zeros(baseBands, targetFrames, 1, Nwin, 'single');

for i = 1:Nwin
    a = (i-1)*hopWin + 1;
    b = a + winLen - 1;
    seg = x(a:b);

    % mel-spectrogram (prefer AT; fallback to STFT+AFB)
    logMel = computeLogMel(seg, sr, win, overlapLen, baseBands, fr);

    % shape normalize to [baseBands x targetFrames]
    [r,c] = size(logMel);
    if r ~= baseBands && c == baseBands
        logMel = logMel.'; [r,c] = size(logMel);
    end
    if r ~= baseBands
        error('slidingWindowScores: unexpected mel shape %dx%d (expected %dx*)', r, c, baseBands);
    end

    Z = zeros(baseBands, targetFrames, 'single');
    if c <= targetFrames
        Z(:,1:c) = logMel;
    else
        startCol = floor((c - targetFrames)/2) + 1;
        Z(:,:) = logMel(:, startCol:startCol+targetFrames-1);
    end

    X(:,:,1,i) = Z;
end

% Global z-score (match extractFeatures behavior)
mu = mean(X(:)); sd = std(X(:)) + eps;
X = (X - mu) / sd;

% ---------- Inference ----------
scores = predict(net, X);      % [Nwin x C]

S = struct();
S.sr         = sr;
S.t_center   = tCenters(:);
S.winSpanSec = windowMs/1000;
S.scores     = scores;
S.classNames = classNames;
end

% ----------------- local helpers -----------------
function classNames = getNetClasses(net)
    classNames = {};
    try
        classNames = cellstr(string(net.Layers(end).Classes));
    catch
        if isprop(net,'Classes')
            classNames = cellstr(string(net.Classes));
        end
    end
    classNames = strtrim(classNames);
end

function fr = chooseDefaultFreqRange(sr)
    nyq = sr/2;
    fr = [50, min(7000, nyq*0.999)];   % your ALL/default baseline
end

function w = localHamming(N)
    try
        w = hamming(N,'periodic');
    catch
        w = hamming(N);
    end
end

function Mlog = computeLogMel(x, sr, win, overlapLen, numBands, fr)
    Mlog = [];
    if exist('melSpectrogram','file') == 2
        try
            M = melSpectrogram(x, sr, 'Window',win, 'OverlapLength',overlapLen, ...
                               'NumBands', numBands, 'FrequencyRange', fr);
            Mlog = log10(M + eps);
            return;
        catch
            try
                M = melSpectrogram(x, sr, 'Window',win, 'OverlapLength',overlapLen, ...
                                   'NumBands', numBands);
                Mlog = log10(M + eps);
                return;
            catch
            end
        end
    end
    % Fallback: STFT + Auditory FB
    if exist('designAuditoryFilterBank','file') ~= 2
        error('Audio Toolbox function "designAuditoryFilterBank" not found.');
    end
    S = spectrogram(x, win, overlapLen, numel(win), sr);
    fb = designAuditoryFilterBank(sr, 'NumBands',numBands, 'FFTLength',numel(win), ...
                                  'FrequencyRange', fr);
    Mlog = log10(fb * abs(S) + eps);
end

function v = getf(cfg, group, name, def)
    v = def;
    if ~isstruct(cfg), return; end
    if isfield(cfg, group)
        g = cfg.(group);
        if isfield(g, name) && ~isempty(g.(name)), v = g.(name); end
    end
end
