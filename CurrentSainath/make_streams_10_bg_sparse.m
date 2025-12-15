function make_streams_10h_bg_sparse()
% make_streams_10h_bg_sparse
% Build ~10 hours per condition with sparse keywords and realistic background.
% CLEAN: quiet room-tone bed so negative scores vary (no flat ~0.07 shelf).
% NOISY: background bed with time-varying SNR across 30s chunks.
%
% Outputs:
%   ./streams_10h_bg_sparse/streams_10h_bg_sparse_clean.mat
%   ./streams_10h_bg_sparse/streams_10h_bg_sparse_noisy.mat
%
% Requires (from your project):
%   kws_config('sainath14'), loadAudioData(), addStreamingWindows()

    cfg = kws_config('sainath14');

    % ---- Duration: 100 x 6 min ~ 10 hours per condition ----
    numStreams   = 100;    % increase if you want >10h
    streamLenSec = 360;    % 6 minutes

    % ---- Sainath geometry: 25/10 ms, 23L+1+8R -> ~335 ms span ----
    frameMs      = getf(cfg,'features','frameMs',25);
    hopMs        = getf(cfg,'features','hopMs',10);
    L            = getf(cfg,'sainath','leftCtx',23);
    R            = getf(cfg,'sainath','rightCtx',8);
    spanMs       = frameMs + hopMs*(L+R);
    hopWinMs     = 10;       % decisions every 10 ms

    % ---- Behavior dials (tweak if needed) ----
    kwPerMin        = 1;         % sparse keyword rate
    minGapSec       = 1.5;       % spacing around events
    tolMs           = 100;       % tighter than 200 ms, keeps P% reasonable
    idleProb        = 0.7;       % chance to insert idle stretch
    idleDurSecRange = [5 15];    % idle stretch length (sec)

    % CLEAN background (quiet room tone) and NOISY controls
    cleanBgRMS      = 0.03;      % 0.02–0.05 is good: quiet but nonzero
    noisyBedScale   = 0.30;      % baseline bed scale for noisy
    noisySNRchoices = [0 5 10 15]; % dB per 30s chunk (varied)

    % ---- Data (from your pipeline) ----
    [~, ~, testFiles, testLabels] = loadAudioData();

    baseDir = fullfile(pwd, 'streams_10h_bg_sparse');
    if ~exist(baseDir,'dir'), mkdir(baseDir); end

    % ---------------- CLEAN ----------------
    cfgC = cfg;
    cfgC.streaming = struct();
    cfgC.streaming.numStreams     = numStreams;
    cfgC.streaming.streamLenSec   = streamLenSec;
    cfgC.streaming.winSpanMs      = spanMs;
    cfgC.streaming.hopWinMs       = hopWinMs;
    cfgC.streaming.keywordsPerMin = kwPerMin;
    cfgC.streaming.minGapSec      = minGapSec;
    cfgC.streaming.noiseSNRdB     = [];         % marks CLEAN
    cfgC.streaming.bgGain         = cleanBgRMS;  % quiet room tone target
    cfgC.sainath.labelTolMs       = tolMs;

    wavCleanDir = fullfile(baseDir,'clean_wav');
    if ~exist(wavCleanDir,'dir'), mkdir(wavCleanDir); end

    streams_clean = makeStreamingCorpus_sparse(cfgC, testFiles, testLabels, wavCleanDir, ...
                        idleProb, idleDurSecRange, false, cleanBgRMS, noisyBedScale, noisySNRchoices);
    streams_clean = addStreamingWindows(streams_clean, cfgC);

    % ensure wavPath points to real files
    for s=1:numel(streams_clean)
        if ~isfile(streams_clean(s).wavPath)
            streams_clean(s).wavPath = fullfile(wavCleanDir, sprintf('stream_%02d.wav', s));
        end
    end

    save(fullfile(baseDir,'streams_10h_bg_sparse_clean.mat'),'streams_clean','-v7.3');
    fprintf('make_streams_10h_bg_sparse: wrote CLEAN -> %s\n', fullfile(baseDir,'streams_10h_bg_sparse_clean.mat'));

    % ---------------- NOISY ----------------
    wavNoisyDir = fullfile(baseDir,'noisy_wav');
    if ~exist(wavNoisyDir,'dir'), mkdir(wavNoisyDir); end

    streams_noisy = repmat(streams_clean(1), numStreams, 1);
    for s = 1:numStreams
        tmp = makeStreamingCorpus_sparse(cfgC, testFiles, testLabels, wavNoisyDir, ...
                    idleProb, idleDurSecRange, true, cleanBgRMS, noisyBedScale, noisySNRchoices);
        tmp = addStreamingWindows(tmp, cfgC);
        tmp(1).wavPath = fullfile(wavNoisyDir, sprintf('stream_%02d.wav', s));
        streams_noisy(s) = tmp(1);
    end

    save(fullfile(baseDir,'streams_10h_bg_sparse_noisy.mat'),'streams_noisy','-v7.3');
    fprintf('make_streams_10h_bg_sparse: wrote NOISY -> %s\n', fullfile(baseDir,'streams_10h_bg_sparse_noisy.mat'));
end

% ======================================================================
% Internal builder: creates one set of streams with idle stretches and
% either quiet CLEAN bed or time-varying-SNR NOISY bed. All helpers below.
function streams = makeStreamingCorpus_sparse(cfg, testFiles, testLabels, outWavDir, idleProb, idleDurSecRange, isNoisy, cleanBgRMS, noisyBedScale, noisySNRchoices)
    if ~exist(outWavDir,'dir'), mkdir(outWavDir); end
    sr = 16000;

    numStreams   = getf(cfg,'streaming','numStreams',5);
    streamLenSec = getf(cfg,'streaming','streamLenSec',60);

    kwPerMin   = getf(cfg,'streaming','keywordsPerMin',1);
    minGapSec  = getf(cfg,'streaming','minGapSec',1.5);

    frameMs = getf(cfg,'features','frameMs',25);
    hopMs   = getf(cfg,'features','hopMs',10);
    L       = getf(cfg,'sainath','leftCtx',23);
    R       = getf(cfg,'sainath','rightCtx',8);
    spanMs  = getf(cfg,'streaming','winSpanMs', frameMs + hopMs*(L+R));
    hopWinMs= getf(cfg,'streaming','hopWinMs',10);

    % Target keywords
    if isfield(cfg,'sainath') && isfield(cfg.sainath,'targetWords') && ~isempty(cfg.sainath.targetWords)
        targetWords = string(lower(cfg.sainath.targetWords(:)'));
    elseif isfield(cfg,'warden') && isfield(cfg.warden,'targetWords') && ~isempty(cfg.warden.targetWords)
        targetWords = string(lower(cfg.warden.targetWords(:)'));
    else
        cats = string(lower(categories(testLabels)));
        targetWords = cats(~startsWith(cats,"_"));
    end

    % Background pool from dataset (if available)
    noiseList = {};
    if isfield(cfg,'paths') && isfield(cfg.paths,'datasetRoot')
        nd = fullfile(cfg.paths.datasetRoot,'_background_noise_');
        if exist(nd,'dir')
            d = dir(fullfile(nd,'*.wav'));
            if ~isempty(d), noiseList = fullfile({d.folder},{d.name}); end
        end
    end
    useDatasetNoise = ~isempty(noiseList);

    [label2files, labelsPresent] = indexByLabel(testFiles, testLabels);

    streams = repmat(struct('wavPath','','fs',sr,'events',table(), ...
                            'winTimesMs',[],'winLabels',categorical(), ...
                            'winStartIdx',[],'winEndIdx',[]), numStreams,1);

    rng(1234);  % reproducible

    for s = 1:numStreams
        Lsamp = round(streamLenSec * sr);
        y     = zeros(Lsamp,1,'single');

        % ---- Background bed in 30s chunks (quiet for CLEAN; SNR-varied for NOISY) ----
        segLen = 30;
        t0 = 0;
        while t0 < streamLenSec
            t1 = min(streamLenSec, t0 + segLen);
            p0 = round(t0*sr)+1; p1 = round(t1*sr);
            len = p1-p0+1;

            if useDatasetNoise
                f = noiseList{randi(numel(noiseList))};
                bed = readMono(f, sr);
                if isempty(bed), bed = pink(len); else, bed = tileToLen(bed, len); end
            else
                % Fallback room-tone if dataset noise folder missing
                bed = pink(len);
            end

            % Normalize RMS -> scale by mode
            r = rms(bed); if r > 0, bed = bed / r; end
            if isNoisy
                snr = noisySNRchoices(randi(numel(noisySNRchoices)));
                bed = bed * (10^(-snr/20)) * noisyBedScale;
            else
                bed = bed * cleanBgRMS;  % quiet but nonzero
            end

            y(p0:p1) = y(p0:p1) + bed;
            t0 = t1;
        end

        % ---- Sparse keywords with optional idle stretches ----
        events = [];
        lambda = max(1e-3, kwPerMin/60);  % per second
        t = 0;
        while true
            gap = exprnd(1/lambda);
            gap = max(gap, minGapSec);
            if rand < idleProb
                idleDur = idleDurSecRange(1) + rand*(diff(idleDurSecRange));
                gap = gap + idleDur;
            end
            t = t + gap;
            onset = round(t * sr);
            if onset >= Lsamp-1, break; end

            lab = chooseKW(targetWords, labelsPresent);
            fKW = pickFile(label2files, lab); if isempty(fKW), continue; end
            xKW = readMono(fKW, sr); if isempty(xKW), continue; end
            xKW = trimZeros(xKW);

            len = min(numel(xKW), Lsamp-onset+1);
            if len <= 10, continue; end

            seg = 0.7 * xKW(1:len);  % mild gain
            y(onset:onset+len-1) = y(onset:onset+len-1) + seg;

            events = [events; onset, onset+len-1, string(lower(lab))]; %#ok<AGROW>

            t = (onset+len-1)/sr + minGapSec;
        end

        % ---- Peak-normalize and save ----
        peak = max(1e-6, max(abs(y)));
        y = (0.95/peak) * y;

        outPath = fullfile(outWavDir, sprintf('stream_%02d.wav', s));
        audiowrite(outPath, y, sr);

        % Pre-fill window geometry (labels added by addStreamingWindows)
        Tms = streamLenSec*1000;
        halfWin = spanMs/2;
        centers = (halfWin):hopWinMs:(Tms-halfWin);
        if isempty(centers), centers = halfWin; end
        starts = centers - halfWin; ends = centers + halfWin;

        streams(s).wavPath     = outPath;
        streams(s).fs          = sr;
        streams(s).events      = toEventTable(events, sr);
        streams(s).winTimesMs  = [starts(:) ends(:)];
        streams(s).winLabels   = categorical(); % will be filled by addStreamingWindows
        streams(s).winStartIdx = max(1, round((starts(:)/1000)*sr));
        streams(s).winEndIdx   = max(streams(s).winStartIdx, round((ends(:)/1000)*sr));
    end
end

% ============================== Helpers ==============================
function T = toEventTable(events, sr)
    if isempty(events)
        T = table('Size',[0 3],'VariableTypes',{'double','double','string'}, ...
                  'VariableNames',{'onset_s','offset_s','label'});
    else
        T = array2table(events, 'VariableNames',{'onset_samp','offset_samp','label'});
        T.onset_s  = double(T.onset_samp)  / sr;
        T.offset_s = double(T.offset_samp) / sr;
        T = removevars(T, {'onset_samp','offset_samp'});
    end
end

function x = readMono(f, sr)
    try
        [x, fs] = audioread(f);
        if size(x,2) > 1, x = mean(x,2); end
        if fs ~= sr, x = resample(x, sr, fs); end
        x = single(max(-1,min(1,x)));
    catch
        x = [];
    end
end

function y = tileToLen(x, L)
    if numel(x) >= L, y = x(1:L); return; end
    reps = ceil(L/numel(x));
    y = repmat(x, reps, 1); y = y(1:L);
end

function y = pink(N)
    % cheap pink-ish noise (fallback room tone)
    white = randn(N,1,'single') * 1e-3;
    b = [0.049922035 -0.095993537 0.050612699 -0.004408786];
    a = [1 -2.494956002 2.017265875 -0.522189400];
    y = filter(b,a,white);
end

function x = trimZeros(x)
    % trim leading/trailing near-zeros to avoid degenerate very-short events
    x = double(x);
    thr = 0.002;
    i1 = find(abs(x) > thr, 1, 'first');
    if isempty(i1), x = single(x); return; end
    i2 = find(abs(x) > thr, 1, 'last');
    x = single(x(i1:i2));
end

function [map, labels] = indexByLabel(files, labelsCat)
    files = files(:);
    if ~iscategorical(labelsCat), labelsCat = categorical(labelsCat); end
    labels = categories(labelsCat);
    map = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(files)
        lab = char(lower(string(labelsCat(i))));
        if ~isKey(map,lab), map(lab) = {}; end
        map(lab) = [map(lab); files{i}];
    end
end

function f = pickFile(map, lab)
    f = '';
    k = char(lower(string(lab)));
    if ~isKey(map,k), return; end
    L = map(k); if isempty(L), return; end
    f = L{randi(numel(L))};
end

function lab = chooseKW(targetWords, labelsPresent)
    ts = intersect(string(lower(targetWords)), string(lower(labelsPresent)), 'stable');
    if isempty(ts), ts = string(lower(labelsPresent)); end
    lab = ts(randi(numel(ts)));
end

function v = getf(S, group, name, def)
    v = def;
    if ~isstruct(S) || ~isfield(S, group), return; end
    G = S.(group);
    if isfield(G, name) && ~isempty(G.(name)), v = G.(name); end
end
