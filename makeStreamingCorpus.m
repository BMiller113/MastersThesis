function streams = makeStreamingCorpus(cfg, testFiles, testLabels, outWavDir)
% makeStreamingCorpus  Build long streaming WAVs + annotations from test clips.
% Outputs a struct array with fields:
%   .wavPath   : full path to the saved stream WAV
%   .fs        : sample rate (16 kHz)
%   .events    : table with [onset_s, offset_s, label] for inserted keywords
%   .winTimesMs: [N x 2] start/end (ms) for each sliding decision window  
%   .winLabels : [N x 1] categorical label per window                    
%

% 10/5 version


% ---------- Configs, defaults ----------
if nargin < 4 || isempty(outWavDir)
    outWavDir = fullfile(pwd,'streams');
end
if ~exist(outWavDir,'dir'), mkdir(outWavDir); end

sr = 16000;  % target sample rate

% Streaming controls
numStreams     = getfield_def(cfg,'streaming','numStreams',        5);  
streamLenSec   = getfield_def(cfg,'streaming','streamLenSec',     60);
kwPerMinute    = getfield_def(cfg,'streaming','keywordsPerMin',   15);   % avg keywords/min
minGapSec      = getfield_def(cfg,'streaming','minGapSec',       0.25);  % min spacing between events
bgGain         = getfield_def(cfg,'streaming','bgGain',          0.3);   % 0..1 background level
kwGain         = getfield_def(cfg,'streaming','kwGain',          0.9);   % pre-normalization keyword gain

% Decision windowing (what the detector sees per decision)
hopWinMs       = getfield_def(cfg,'streaming','hopWinMs',        100);   % decisions every 100 ms
winSpanMs      = getfield_def(cfg,'streaming','winSpanMs',       500);   % each decision sees a 500 ms span
labelTolMs     = getfield_def(cfg,'sainath','labelTolMs',        100);   % allow +/- tolerance around events

% Target keyword list: use cfg.warden.targetWords if present; otherwise infer
if isfield(cfg,'warden') && isfield(cfg.warden,'targetWords') && ~isempty(cfg.warden.targetWords)
    targetWords = string(cfg.warden.targetWords(:)');
else
    cats = string(categories(testLabels));
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
                        'winTimesMs',[],'winLabels',categorical()), numStreams,1);

for s = 1:numStreams
    L = round(streamLenSec * sr);
    y = zeros(L,1,'single');

    % Background bed
    if ~isempty(noiseList)
        y = y + bgGain * stitchBackground(noiseList, L, sr);
    end

    % Insert keyword events with approximate Poisson rate
    expGapSec = 60 / max(1,kwPerMinute);        
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

        % Read keyword audio
        x = readWavMono(f, sr);
        if isempty(x), continue; end
        % Light random trim of leading/trailing silence (optional)
        x = trimZeros(x);

        % Mix in at onset (with simple bounds check)
        len = numel(x);
        if onset+len-1 > L
            len = L - onset + 1;
        end
        if len <= 10, continue; end

        seg = x(1:len);
        % Scale keyword relative to bed
        seg = kwGain * seg;

        % Add into stream (accumulate)
        y(onset:onset+len-1) = y(onset:onset+len-1) + seg;

        % Record event (keep label in lower-case to match class list)
        events = [events; onset onset+len-1 string(lower(lab))]; %#ok<AGROW>

        % Respect minimum gap after event
        t = (onset+len-1)/sr + minGapSec;
    end

    % Peak normalize, matlab avoid audiowrite "clipped" warnings
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

    % ---------- aquire sliding decision windows + labels ----------
    % 10/5 version
    Tms = streamLenSec*1000;
    halfWin = winSpanMs/2;
    centers = (halfWin):hopWinMs:(Tms-halfWin);
    if isempty(centers)
        centers = halfWin;
    end
    starts = centers - halfWin;
    ends   = centers + halfWin;
    winTimesMs = [starts(:) ends(:)];

    % Label each window: if it overlaps any event (with tolerance), assign that keyword.
    % If multiple overlap, pick the one with max overlap; else '_neg_'.
    winLabels = strings(numel(centers),1);
    if ~isempty(E)
        evOn = 1000*E.onset_s;   % ms
        evOff= 1000*E.offset_s;  % ms
        evLab= string(E.label);
        for w = 1:numel(centers)
            a = starts(w) - labelTolMs;
            b = ends(w)   + labelTolMs;
            % overlap test: max(0, min(b,evOff)-max(a,evOn))
            ov = max(0, min(b, evOff) - max(a, evOn));
            if any(ov > 0)
                [~,k] = max(ov);
                winLabels(w) = lower(evLab(k));
            else
                winLabels(w) = "_neg_";
            end
        end
    else
        winLabels(:) = "_neg_";
    end

    % Pack stream
    streams(s).wavPath   = outPath;
    streams(s).fs        = sr;
    streams(s).events    = E;
    streams(s).winTimesMs= winTimesMs;
    streams(s).winLabels = categorical(winLabels);
end
end

% -----------------helpers--------------------

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
        lab = char(labelsCat(i));
        if ~isKey(map, lab), map(lab) = {}; end
        map(lab) = [map(lab); files{i}];
    end
end

function f = pickRandomFile(map, lab)
    f = '';
    k = char(lab);
    if ~isKey(map,k), return; end
    L = map(k);
    if isempty(L), return; end
    f = L{randi(numel(L))};
end

function lab = chooseKeyword(targetWords, labelsPresent)
    % pick from intersection of desired targets and labels we actually have
    ts = intersect(string(targetWords), string(labelsPresent), 'stable');
    if isempty(ts), ts = labelsPresent; end
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
        % hard clip safety (should be rare with good sources)
        x = max(-1,min(1,x));
        x = single(x);
    catch
        x = [];
    end
end

function y = trimZeros(x)
    % trim leading/trailing 0s, could maybe be improved
    x = double(x);
    thr = 0.002;
    nz = find(abs(x) > thr, 1, 'first');
    if isempty(nz), y = single(x); return; end
    nz2 = find(abs(x) > thr, 1, 'last');
    y = single(x(nz:nz2));
end
