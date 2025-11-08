function ev = evaluateSainathEvents(net, streams, cfg)
%EVALUATESAINATHEVENTS  Event-level FR(%) vs FA/hour with smoothing + hangover + NMS.
% Output struct:
%   ev.thr        : [Kx1] thresholds
%   ev.FA_per_hr  : [Kx1] false alarms per hour (events/hour)
%   ev.FR_percent : [Kx1] false rejects (%) at each threshold

% ---------- Params (with safe defaults) ----------
hangoverSec   = ev_getf(cfg,'sainath','event','hangoverSec',   0.50);  % merge adjacent hits
minSepSec     = ev_getf(cfg,'sainath','event','minSepSec',     0.75);  % NMS separation
matchTolMs    = ev_getf(cfg,'sainath','event','matchTolMs',      500); % GT match tolerance
minDetDurSec  = ev_getf(cfg,'sainath','event','minDetDurSec',  0.10);  % min on-duration
smoothWinSec  = ev_getf(cfg,'sainath','event','smoothWinSec',  0.05);  % moving avg smoothing (sec)
thrLo         = ev_getf(cfg,'sainath','event','thrLo',         0.00);  % broaden thr range
thrHi         = ev_getf(cfg,'sainath','event','thrHi',         1.00);
thrN          = ev_getf(cfg,'sainath','event','thrN',            801); % enough resolution

% ---------- Gather per-stream scores and metadata ----------
[scoreCell, timeCell, durSec] = ev_get_stream_scores(net, streams, cfg, smoothWinSec);

% Build threshold grid
thr = linspace(thrLo, thrHi, thrN).';

% Ground-truth events per stream
gt = ev_get_gt_events(streams);

% Tally across streams
totalHours = sum(durSec) / 3600;
K = numel(thr);
FA_per_hr  = zeros(K,1);
FR_percent = zeros(K,1);

for k = 1:K
    th = thr(k);
    FA_count = 0;
    miss_count = 0;
    pos_count = 0;

    for s = 1:numel(streams)
        t = timeCell{s};          % [Nw x 1] time centers (sec)
        p = scoreCell{s};         % [Nw x 1] trigger score (max over KW classes)

        % Binary sequence above threshold
        b = (p >= th);

        % Hangover (dilation)
        b = ev_apply_hangover(b, t, hangoverSec);

        % Segment -> detections (time of max score per segment)
        detTimes = ev_segments_to_detections(b, p, t, minDetDurSec);

        % Non-maximum suppression: enforce min separation
        detTimes = ev_enforce_min_sep(detTimes, minSepSec);

        % Match detections to GT (any keyword)
        [FA_s, TP_s, miss_s, pos_s] = ev_match_events(detTimes, gt{s}, matchTolMs/1000);

        FA_count   = FA_count   + FA_s;
        miss_count = miss_count + miss_s;
        pos_count  = pos_count  + pos_s;
    end

    % Aggregate metrics
    if totalHours <= 0, totalHours = 1e-12; end
    FA_per_hr(k) = FA_count / totalHours;
    if pos_count == 0
        FR_percent(k) = 100;
    else
        FR_percent(k) = (miss_count / pos_count) * 100;
    end
end

% Keep only finite rows (avoid plotting headaches)
good = isfinite(FA_per_hr) & isfinite(FR_percent);
thr        = thr(good);
FA_per_hr  = FA_per_hr(good);
FR_percent = FR_percent(good);

ev = struct('thr',thr, 'FA_per_hr',FA_per_hr, 'FR_percent',FR_percent);
end

% ==================== Local helpers (namespaced) ====================

function [scoreCell, timeCell, durSec] = ev_get_stream_scores(net, streams, cfg, smoothWinSec)
% Returns per-stream:
%   scoreCell{s} : [Nw x 1] trigger score (max over keyword classes, smoothed)
%   timeCell{s}  : [Nw x 1] time centers (sec)
%   durSec(s)    : stream duration (sec)

    classNames = ev_tryGetClasses(net);

    % Keyword list: from cfg.sainath.targetWords if set, else from classes
    if isfield(cfg,'sainath') && isfield(cfg.sainath,'targetWords') && ~isempty(cfg.sainath.targetWords)
        tgt = lower(string(cfg.sainath.targetWords(:)'));
    else
        tgt = string(classNames(:)');
        tgt = tgt(~startsWith(tgt,"_"));
    end

    scoreCell = cell(numel(streams),1);
    timeCell  = cell(numel(streams),1);
    durSec    = zeros(numel(streams),1);

    for i = 1:numel(streams)
        X4 = ev_featuresFromWavAndWindows(streams(i), cfg);  % [H x W x 1 x Nw]
        S  = predict(net, X4);                                % [Nw x C]

        % time centers from winTimesMs
        Tms  = mean(streams(i).winTimesMs, 2);
        tsec = Tms / 1000;

        % trigger = max over keyword classes
        idxKW = false(1,size(S,2));
        for j=1:numel(tgt)
            jj = find(strcmpi(classNames, tgt(j)), 1);
            if ~isempty(jj), idxKW(jj) = true; end
        end
        if ~any(idxKW)
            % fallback: take max over all non-underscore classes
            idxKW = ~startsWith(string(classNames), "_");
        end
        pKW = max(S(:, idxKW), [], 2);  % [Nw x 1]

        % temporal smoothing (moving average over ~smoothWinSec)
        if numel(tsec) >= 3 && smoothWinSec > 0
            dt = median(diff(tsec)); if ~isfinite(dt) || dt<=0, dt = 0.01; end
            w  = max(1, round(smoothWinSec / dt));
            if w > 1
                pKW = movmean(pKW, [w-1, 0]);  % causal smoothing
            end
        end

        scoreCell{i} = pKW(:);
        timeCell{i}  = tsec(:);

        info = audioinfo(streams(i).wavPath);
        durSec(i) = info.Duration;
    end
end

function classNames = ev_tryGetClasses(net)
    try
        classNames = cellstr(string(net.Layers(end).Classes));
    catch
        classNames = {'_neg_'};
    end
end

function X4 = ev_featuresFromWavAndWindows(s, cfg)
    % Build features for each decision window in stream s.
    srTgt = ev_getf(cfg,'features','','forceSampleRate',16000);
    [x, fs] = audioread(s.wavPath);
    if size(x,2) > 1, x = mean(x,2); end
    if fs ~= srTgt, x = resample(x, srTgt, fs); fs = srTgt; end

    starts = s.winStartIdx(:); ends = s.winEndIdx(:);
    starts = max(starts,1); ends = min(ends, numel(x)); ends = max(ends, starts);

    tmpDir = tempname; mkdir(tmpDir);
    c = onCleanup(@() ev_safeRmDir(tmpDir));
    nW = numel(starts);
    pathsTmp = cell(nW,1);
    for j = 1:nW
        seg = x(starts(j):ends(j));
        mx  = max(abs(seg)); if mx>1, seg = seg./mx; end
        p = fullfile(tmpDir, sprintf('win_%06d.wav', j));
        audiowrite(p, seg, fs);
        pathsTmp{j} = p;
    end

    % Use your existing extractor
    X4 = extractFeatures(pathsTmp, 'all', 'default', cfg);
end

function gt = ev_get_gt_events(streams)
    gt = cell(numel(streams),1);
    for i = 1:numel(streams)
        E = streams(i).events;
        if isempty(E)
            gt{i} = struct('onset',[],'offset',[],'label',strings(0,1));
        else
            gt{i} = struct('onset', E.onset_s(:), ...
                           'offset', E.offset_s(:), ...
                           'label', lower(string(E.label(:))));
        end
    end
end

function b2 = ev_apply_hangover(b, t, hangoverSec)
    if ~any(b), b2 = b; return; end
    b = b(:); t = t(:);
    dt = median(diff(t)); 
    if ~isfinite(dt) || dt<=0, dt = 0.01; end
    n = max(1, round(hangoverSec / dt));
    k = ones(2*n+1,1);
    b2 = conv(double(b), k, 'same') > 0;
    b2 = logical(b2);
end

function detTimes = ev_segments_to_detections(b, p, t, minDurSec)
    detTimes = [];
    if ~any(b), return; end
    b = b(:); p = p(:); t = t(:);
    d = diff([0; b; 0]);
    onIdx  = find(d==1);
    offIdx = find(d==-1)-1;
    for i = 1:numel(onIdx)
        a = onIdx(i); z = offIdx(i);
        dur = max(eps, t(z) - t(a));
        if dur < max(minDurSec, 0), continue; end
        [~, m] = max(p(a:z));
        detTimes(end+1,1) = t(a + m - 1); %#ok<AGROW>
    end
end

function detTimes = ev_enforce_min_sep(detTimes, minSepSec)
    if numel(detTimes) <= 1, return; end
    detTimes = sort(detTimes(:));
    keep = true(size(detTimes));
    last = detTimes(1);
    for i = 2:numel(detTimes)
        if detTimes(i) - last < minSepSec
            keep(i) = false;
        else
            last = detTimes(i);
        end
    end
    detTimes = detTimes(keep);
end

function [FA_s, TP_s, miss_s, pos_s] = ev_match_events(detTimes, gt, tolSec)
    FA_s = 0; TP_s = 0; miss_s = 0; pos_s = 0;
    pos_s = numel(gt.onset);
    if pos_s == 0
        FA_s = numel(detTimes);
        return;
    end
    gtCenters = (gt.onset(:) + gt.offset(:))/2;
    used = false(size(gtCenters));
    for i = 1:numel(detTimes)
        t = detTimes(i);
        [dmin, j] = min(abs(gtCenters - t)); %#ok<ASGLU>
        if ~isempty(j) && dmin <= tolSec && ~used(j)
            TP_s = TP_s + 1;
            used(j) = true;
        else
            FA_s = FA_s + 1;
        end
    end
    miss_s = sum(~used);
end

function v = ev_getf(cfg, group, subgroup, name, def)
    % Safe getter for nested cfg fields: cfg.(group).(subgroup).(name)
    v = def;
    if ~isstruct(cfg), return; end
    if ~isfield(cfg, group), return; end
    G = cfg.(group);
    if ~isempty(subgroup)
        if ~isfield(G, subgroup), return; end
        S = G.(subgroup);
        if isfield(S, name) && ~isempty(S.(name)), v = S.(name); end
    else
        if isfield(G, name) && ~isempty(G.(name)), v = G.(name); end
    end
end

function ev_safeRmDir(d)
    if exist(d,'dir'), try, rmdir(d,'s'); catch, end, end
end
