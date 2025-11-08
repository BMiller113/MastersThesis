function streams = makeStreamingCorpus(cfg, testFiles, testLabels, outWavDir)
% makeStreamingCorpus  Build long streaming WAVs + annotations from test clips.
% Outputs a struct array with fields:
%   .wavPath    : full path to the saved stream WAV
%   .fs         : sample rate (16 kHz)
%   .events     : table with [onset_s, offset_s, label] for inserted keywords
%   .winTimesMs : [N x 2] start/end (ms) for each sliding decision window
%   .winLabels  : [N x 1] categorical label per window
%   .winStartIdx, .winEndIdx : sample indices for each window (fast slicing)
%
% Notes:
% - Audio is peak-normalized after mixing.
% - “Noisy” mixing may be controlled by cfg.streaming.noiseSNRdB (e.g., 10).
%   If set, the background bed is scaled to reach the requested SNR (dB)
%   relative to the keyword mixture; otherwise cfg.streaming.bgGain is used.

% ---------- Inputs / defaults ----------
if nargin < 4 || isempty(outWavDir)
    outWavDir = fullfile(pwd,'streams');
end
if ~exist(outWavDir,'dir'), mkdir(outWavDir); end

sr = 16000;  % target sample rate

% Streaming controls (cfg.streaming with fallbacks)
numStreams     = getfield_def(cfg,'streaming','numStreams',        5);
streamLenSec   = getfield_def(cfg,'streaming','streamLenSec',     60);
kwPerMinute    = getfield_def(cfg,'streaming','keywordsPerMin',   15);   % avg keywords/min
minGapSec      = getfield_def(cfg,'streaming','minGapSec',       0.25);  % min spacing between events

% Decision/windowing for labeling (Sainath-style uses 100 ms hop)
hopWinMs       = getfield_def(cfg,'streaming','hopWinMs',        100);
% If you're matching Sainath stacking (25 ms + 10 ms * (23+8)), span ≈ 325 ms.
% We'll honor cfg.streaming.winSpanMs if provided, else derive from features:
spanFromFeatMs = getfield_def(cfg,'features','frameMs',25) + ...
                 getfield_def(cfg,'features','hopMs',10) * ...
                 (getfield_def(cfg,'sainath','leftCtx',23) + getfield_def(cfg,'sainath','rightCtx',8));
winSpanMs      = getfield_def(cfg,'streaming','winSpanMs', round(spanFromFeatMs));
labelTolMs     = getfield_def(cfg,'sainath',  'labelTolMs', 100);  % +/- tolerance

% Mixing controls
bgGain         = getfield_def(cfg,'streaming','bgGain',          0.3);   % used if noiseSNRdB is empty
noiseSNRdB     = getfield_def(cfg,'streaming','noiseSNRdB',      []);    % set (e.g. 10) to force SNR mixing

% Keyword gain (pre-normalization)
kwGain         = getfield_def(cfg,'streaming','kwGain',          0.9);

% Target keyword list: prefer cfg.sainath.targetWords, else infer from labels
if isfield(cfg,'sainath') && isfield(cfg.sainath,'targetWords') && ~isempty(cfg.sainath.targetWords)
    targetWords = string(lower(cfg.sainath.targetWords(:)'));
elseif isfield(cfg,'warden') && isfield(cfg.warden,'targetWords') && ~isempty(cfg.warden.targetWords)
    targetWords = string(lower(cfg.warden.targetWords(:)'));
else
    cats = string(lower(categories(testLabels)));
    targetWords = cats(~startsWith(cats,"_"));  % drop _unknown_, _silence_ if present
end

% Background noise pool (Speech Commands style)
noiseDir = '';
if isfield(cfg,'paths') && isfield(cfg.paths,'datasetRoot')
    try
        noiseDir = fullfile(cfg.paths.datasetRoot,'_background_noise_');
    catch, noiseDir = ''; end
end
noiseList = {};
if ~isempty(noiseDir) && exist(noiseDir,'dir')
    d = dir(fullfile(noiseDir,'*.wav'));
    noiseList = fullfile({d.folder},{d.name});
end

% Index test files by label for quick sampling
[label2files, labelsPresent] = indexByLabel(testFiles, testLabels);

% ---------- Build streams ----------
streams = repmat(struct('wavPath','','fs',sr,'events',table(), ...
                        'winTimesMs',[],'winLabels',categorical(), ...
                        'winStartIdx',[],'winEndIdx',[]), numStreams,1);

for s = 1:numStreams
    L = round(streamLenSec * sr);

    % We'll accumulate keywords into y_kw (signal), then mix in noise bed "n"
    y_kw = zeros(L,1,'single');

    % Insert keyword events with approximate Poisson rate
    expGapSec = 60 / max(1,kwPerMinute);        % mean inter-arrival
    t = 0;
    events = [];  % [onsetSample offsetSample stringLabel]
    rng(1234 + s); % reproducible per stream

    while true
        % Sample next gap
        gap = exprnd(expGapSec);     % exponential
        gap = max(gap, minGapSec);   % respect minimum gap
        t = t + gap;
        onset = round(t * sr);
        if onset >= L-1
            break;
        end

        % Choose a keyword that exists in test set
        lab = chooseKeyword(targetWords, labelsPresent);
        f   = pickRandomFile(label2files, lab);
        if isempty(f), continue; end

        % Read/cook keyword audio
        x = readWavMono(f, sr);
        if isempty(x), continue; end
        x = trimZeros(x);

        % Mix in at onset (with simple bounds check)
        len = numel(x);
        if onset+len-1 > L
            len = L - onset + 1;
        end
        if len <= 10, continue; end

        seg = kwGain * x(1:len);
        y_kw(onset:onset+len-1) = y_kw(onset:onset+len-1) + seg;

        % Record event (lower-case to match class list)
        events = [events; onset onset+len-1 string(lower(lab))]; %#ok<AGROW>

        % Respect minimum gap after event
        t = (onset+len-1)/sr + minGapSec;
    end

    % --- Build an unscaled background bed "n" of the same length ---
    n = zeros(L,1,'single');
    if ~isempty(noiseList)
        n = stitchBackground(noiseList, L, sr);
    end

    % --- Mixing: SNR-controlled if noiseSNRdB is set; else simple gain ---
    if ~isempty(noiseSNRdB) && isfinite(noiseSNRdB)
        rSig  = sqrt(mean(y_kw.^2) + 1e-12);
        rNoi  = sqrt(mean(n.^2) + 1e-12);
        if rNoi > 0 && rSig > 0
            alpha = rSig / (10^(noiseSNRdB/20) * rNoi);
            y = y_kw + alpha * n;
        else
            y = y_kw;  % degenerate case
        end
    else
        y = y_kw + bgGain * n;
    end

    % Peak normalize to avoid audiowrite "clipped" warnings
    peak = max(1e-6, max(abs(y)));
    y = (0.95/peak) * y;

    % Save
    outPath = fullfile(outWavDir, sprintf('stream_%02d.wav', s));
    audiowrite(outPath, y, sr);

    % Build events table (seconds)
    if ~isempty(events)
        E = array2table(events, 'VariableNames',{'onset_samp','offset_samp','label'});
        E.onset_s = double(E.onset_samp) / sr;
        E.offset_s = double(E.offset_samp) / sr;
        E = removevars(E, {'onset_samp','offset_samp'});
    else
        E = table('Size',[0 3],'VariableTypes',{'double','double','string'}, ...
                  'VariableNames',{'onset_s','offset_s','label'});
    end

    % ---------- Sliding decision windows + labels ----------
    Tms = streamLenSec*1000;
    halfWin = winSpanMs/2;
    centers = (halfWin):hopWinMs:(Tms-halfWin);
    if isempty(centers)
        centers = halfWin;
    end
    starts = centers - halfWin;
    ends   = centers + halfWin;
    winTimesMs = [starts(:) ends(:)];

    % Label each window: keyword label if any event overlaps (with tol), else '_neg_'
    winLabels = strings(numel(centers),1);
    if ~isempty(E)
        evOn = 1000*E.onset_s;   % ms
        evOff= 1000*E.offset_s;  % ms
        evLab= string(lower(E.label));
        for w = 1:numel(centers)
            a = starts(w) - labelTolMs;
            b = ends(w)   + labelTolMs;
            ov = max(0, min(b, evOff) - max(a, evOn));  % ms overlap
            if any(ov > 0)
                [~,k] = max(ov);
                winLabels(w) = evLab(k);
            else
                winLabels(w) = "_neg_";
            end
        end
    else
        winLabels(:) = "_neg_";
    end

    % Pack stream
    streams(s).wavPath    = outPath;
    streams(s).fs         = sr;
    streams(s).events     = E;
    streams(s).winTimesMs = winTimesMs;
    streams(s).winLabels  = categorical(winLabels);

    % Also cache sample indices for fast feature slicing
    streams(s).winStartIdx = max(1, round((starts(:)/1000) * sr));
    streams(s).winEndIdx   = max(streams(s).winStartIdx, round((ends(:)/1000) * sr));
end
end

% ----------------- helpers --------------------
function v = getfield_def(cfg, group, name, def)
    v = def;
    if ~isstruct(cfg), return; end
    if isfield(cfg, group)
        g = cfg.(group);
        if isfield(g, name) && ~isempty(g.(name)), v = g.(name); end
    end
end

function [map, labels] = indexByLabel(files, labelsCat)
    files = files(:);
    if ~iscategorical(labelsCat), labelsCat = categorical(labelsCat); end
    labels = categories(labelsCat);
    map = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(files)
        lab = char(lower(string(labelsCat(i))));
        if ~isKey(map, lab), map(lab) = {}; end
        map(lab) = [map(lab); files{i}]; 
    end
end

function f = pickRandomFile(map, lab)
    f = '';
    k = char(lower(string(lab)));
    if ~isKey(map,k), return; end
    L = map(k);
    if isempty(L), return; end
    f = L{randi(numel(L))};
end

function lab = chooseKeyword(targetWords, labelsPresent)
    % pick from intersection of desired targets and labels
    ts = intersect(string(lower(targetWords)), string(lower(labelsPresent)), 'stable');
    if isempty(ts), ts = string(lower(labelsPresent)); end
    lab = ts(randi(numel(ts)));
end

function bed = stitchBackground(noiseList, L, sr)
    bed = zeros(L,1,'single');
    p = 1;
    while p <= L
        f = noiseList{randi(numel(noiseList))};
        x = readWavMono(f, sr);
        if isempty(x), x = randn(sr,1)*1e-4; end
        len = min(numel(x), L-p+1);
        bed(p:p+len-1) = bed(p:p+len-1) + x(1:len);
        p = p + len;
    end
    % light RMS normalization of bed (optional)
    r = sqrt(mean(bed.^2) + 1e-12);
    if r > 0, bed = bed / (2*r); end
end

function x = readWavMono(f, sr)
    try
        [x, fs] = audioread(f);
        if size(x,2) > 1, x = mean(x,2); end
        if fs ~= sr, x = resample(x, sr, fs); end
        x = max(-1,min(1,x));
        x = single(x);
    catch
        x = [];
    end
end

function y = trimZeros(x)
    % trim of leading/trailing near-zeros
    x = double(x);
    thr = 0.002;
    nz = find(abs(x) > thr, 1, 'first');
    if isempty(nz), y = single(x); return; end
    nz2 = find(abs(x) > thr, 1, 'last');
    y = single(x(nz:nz2));
end
