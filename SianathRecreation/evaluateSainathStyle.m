function res = evaluateSainathStyle(net, streams, cfg)
% evaluateSainathStyle
% Sainath-style streaming evaluation on a pre-built "streams" corpus.
% - 40-dim log-mel, 25ms window, 10ms hop
% - Stacking: 23 left + 8 right -> 32 frames total
% - Decision at every frame (10ms) unless cfg overrides
% - Threshold-sweeps to compute FR (%) vs FA/hour (x-axis)
%
% Returns:
%   thresholds, fpr, tpr, FR_percent, FA_percent, FA_per_hr, AUC
%   positiveLabel, thr_used, FR_at_op_percent, FAh_at_op
%   faTargets, FR_at_targets, thr_at_targets
%   (optional) CI_AUC, CI_FR, CI_FAh

verbose = false;  % quiet; set true for brief prints

% --- derive scores/labels/classes using Sainath front-end ---
[allScores, allLabels, classNames] = collectScoresAndLabels_Sainath(net, streams, cfg);

% Positive class (forced or dominant non-filler)
posLabel = choosePositiveLabel(cfg, classNames, allLabels);
classIdx = find(strcmp(classNames, posLabel), 1);
if isempty(classIdx), error('Positive label "%s" not found.', posLabel); end

scores = allScores(:, classIdx);                          % Nx1 posterior
y = logical(strcmp(string(allLabels), string(posLabel))); % Nx1 logical
Npos = sum(y); Nneg = sum(~y);
if Npos == 0, warning('No positives for "%s" — TPR=0, FR%%=100.', posLabel); end
if Nneg == 0, error('No negatives in windows — cannot compute FPR/FAh.'); end
Npos = max(1,Npos); Nneg = max(1,Nneg);

% --- threshold grid (auto from score support; dense) ---
if isempty(scores)
    thrGrid = linspace(0,1,2001).';
else
    lo = max(0, min(scores) - 1e-6);
    hi = min(1, max(scores) + 1e-6);
    thrGrid = linspace(lo, hi, 4001).';
end
if isfield(cfg,'sainath') && isfield(cfg.sainath,'thrGrid') && ~isempty(cfg.sainath.thrGrid)
    if numel(cfg.sainath.thrGrid) >= 50
        thrGrid = cfg.sainath.thrGrid(:);
    else
        warning('cfg.sainath.thrGrid has %d point(s) — ignoring.', numel(cfg.sainath.thrGrid));
    end
end

% --- vectorized sweep ---
S = scores(:)'; Y = y(:)'; T = thrGrid(:)';
predPos = S >= T.';                 % KxN
TP = sum(predPos &  Y, 2);
FP = sum(predPos & ~Y, 2);
tpr = TP ./ Npos;
fpr = FP ./ Nneg;
frr = 1 - tpr;

% sort by FPR for stable AUC
[fpr, idx] = sort(fpr, 'ascend');
tpr = tpr(idx); frr = frr(idx); thrGrid = thrGrid(idx);

FR_percent = frr * 100;
FA_percent = fpr * 100;

% --- FA/hour from decision cadence ---
hopMs = getf(cfg,'features','hopMs', 10);      % Sainath = 10 ms per decision
decisionsPerHour = 3600 / (hopMs/1000);       % 360,000 decisions/hour at 10 ms
FA_per_hr = fpr * decisionsPerHour;

% --- AUC, operating point (EER unless fixed) ---
AUC = trapz(fpr, tpr);
if isfield(cfg,'experiments') && ~isempty(cfg.experiments.fixedThreshold)
    thr_used = cfg.experiments.fixedThreshold;
else
    [~, kEER] = min(abs(fpr - (1 - tpr)));
    thr_used = thrGrid(kEER);
end
pred_op       = scores >= thr_used;
FR_at_op_percent = (sum(~pred_op & y)  / max(1,sum(y)))  * 100;
FAh_at_op        = (sum( pred_op & ~y) / max(1,sum(~y))) * decisionsPerHour;

% --- FR at specific FA/h targets (nearest) ---
faTargets = []; FR_at_targets = []; thr_at_targets = [];
if isfield(cfg,'sainath') && isfield(cfg.sainath,'faTargets') && ~isempty(cfg.sainath.faTargets)
    faTargets = cfg.sainath.faTargets(:);
    FR_at_targets  = nan(numel(faTargets),1);
    thr_at_targets = nan(numel(faTargets),1);
    for i = 1:numel(faTargets)
        [~, j] = min(abs(FA_per_hr - faTargets(i)));
        FR_at_targets(i)  = FR_percent(j);
        thr_at_targets(i) = thrGrid(j);
    end
end

% (Optional) bootstrap over streams — unchanged
CI_AUC=[]; CI_FR=[]; CI_FAh=[];
if isfield(cfg,'sainath') && isfield(cfg.sainath,'bootstrapN') && cfg.sainath.bootstrapN > 0
    B = cfg.sainath.bootstrapN;
    [sbcs, ybcs] = splitByStream_Sainath(streams, net, classIdx, posLabel, classNames, cfg);
    Ns = numel(sbcs);
    AUCb = zeros(B,1); FRb = zeros(B,1); FAhb = zeros(B,1);
    for b = 1:B
        pick = randi(Ns, [Ns,1]);
        sb = cell2mat(sbcs(pick)); yb = cell2mat(ybcs(pick));
        [tpr_b, fpr_b, frr_b] = sweepROC(sb, yb, thrGrid);
        AUCb(b) = trapz(fpr_b, tpr_b);
        [~, kb] = min(abs(fpr_b - frr_b));
        thr_b = thrGrid(kb);
        pred_b = sb >= thr_b;
        FRb(b)  = (sum(~pred_b & yb)  / max(1,sum(yb)))   * 100;
        FAhb(b) = (sum( pred_b & ~yb) / max(1,sum(~yb)))  * decisionsPerHour;
    end
    CI_AUC = prctile(AUCb,[2.5 97.5]); CI_FR = prctile(FRb,[2.5 97.5]); CI_FAh = prctile(FAhb,[2.5 97.5]);
end

% --- pack ---
res = struct();
res.thresholds        = thrGrid(:);
res.fpr               = fpr(:);
res.tpr               = tpr(:);
res.FR_percent        = FR_percent(:);
res.FA_percent        = FA_percent(:);
res.FA_per_hr         = FA_per_hr(:);
res.AUC               = AUC;
res.positiveLabel     = posLabel;
res.thr_used          = thr_used;
res.FR_at_op_percent  = FR_at_op_percent;
res.FAh_at_op         = FAh_at_op;
res.faTargets         = faTargets;
res.FR_at_targets     = FR_at_targets;
res.thr_at_targets    = thr_at_targets;
if ~isempty(CI_AUC)
    res.CI_AUC = CI_AUC; res.CI_FR = CI_FR; res.CI_FAh = CI_FAh;
end

if verbose
    fprintf('AUC=%.4f | EER thr=%.4f | FR@op=%.2f%% | FA/h@op=%.3f\n', AUC, thr_used, FR_at_op_percent, FAh_at_op);
end

% ------------------------ helpers ------------------------
function [tpr_, fpr_, frr_] = sweepROC(s_, y_, thr_)
    s_ = s_(:)'; y_ = y_(:)'; thr_ = thr_(:)';
    pred_ = s_ >= thr_.';
    P_ = max(1,sum(y_==1)); N_ = max(1,sum(y_==0));
    TP_ = sum(pred_ &  y_, 2); FP_ = sum(pred_ & ~y_, 2);
    tpr_ = TP_ ./ P_; fpr_ = FP_ ./ N_; frr_ = 1 - tpr_;
end

function [scoresAll, labelsAll, classNames] = collectScoresAndLabels_Sainath(net, streams, cfg)
    % Build 40x32 stacks at every 10ms decision (unless cfg overrides) and score.
    scoresAll = []; labelsAll = categorical();
    classNames = tryGetClasses(net);

    % Front-end (Sainath)
    B = getf(cfg,'features','baseBands',  40);  % 40 mel bins
    W = getf(cfg,'features','targetFrames',32); % 32 stacked frames
    frameMs = getf(cfg,'features','frameMs', 25);
    hopMs   = getf(cfg,'features','hopMs',   10);
    L = getf(cfg,'sainath','leftCtx',  23);
    R = getf(cfg,'sainath','rightCtx', 8);
    if W ~= (L+1+R)
        % keep your requested input width consistent with the stack
        W = (L+1+R);
    end

    for i = 1:numel(streams)
        if ~(isfield(streams(i),'winTimesMs') && isfield(streams(i),'winLabels'))
            error('Stream %d missing winTimesMs/winLabels (rebuild cache).', i);
        end

        % Load audio
        [x, fs] = audioread(streams(i).wavPath);
        if size(x,2) > 1, x = mean(x,2); end
        if fs ~= 16000, x = resample(x,16000,fs); fs = 16000; end
        x = single(max(-1,min(1,x)));

        % Full-stream log-mel frames (B x T)
        [Mlog, Tcenters_ms] = fullLogMelMatrix(x, fs, B, frameMs, hopMs);

        % Decision centers from winTimes (use window centers)
        c_ms = mean(streams(i).winTimesMs,2);      % [Nwin x 1] ms
        % Map each center to nearest frame index
        idxFrames = round((c_ms - Tcenters_ms(1)) / hopMs) + 1;

        % Stack 23L+1+8R around each frame (pad with edge frames)
        Nw = numel(idxFrames);
        X4 = zeros(B, W, 1, Nw, 'single');
        for j = 1:Nw
            c = idxFrames(j);
            left  = max(1, c - L);
            right = min(size(Mlog,2), c + R);
            % extract and pad to W columns
            block = Mlog(:, left:right);
            if size(block,2) < W
                need = W - size(block,2);
                padLeft  = max(0, L - (c-left));
                padRight = need - padLeft;
                if padLeft > 0,  block = [repmat(Mlog(:,left),  1, padLeft),  block]; end
                if padRight > 0, block = [block, repmat(Mlog(:,right), 1, padRight)]; end
            end
            X4(:,:,1,j) = block(:,1:W);
        end

        % Z-score per stream (match your extractor behavior)
        mu = mean(X4(:)); sd = std(X4(:)) + eps;
        X4z = (X4 - mu) / sd;

        S = predict(net, X4z);        % [Nwin x C]
        if isempty(scoresAll), scoresAll = S; else, scoresAll = [scoresAll; S]; end %#ok<AGROW>
        labelsAll = [labelsAll; streams(i).winLabels(:)]; %#ok<AGROW>
    end

    % Constrain label cats within model domain when possible
    if isstring(classNames) || iscellstr(classNames)
        extra = setdiff(categories(labelsAll), cellstr(classNames));
        if ~isempty(extra) && any(strcmp(classNames,"_unknown_"))
            labelsAll = mergecats(labelsAll, extra, "_unknown_");
        end
    end
end

function [sbcs, ybcs] = splitByStream_Sainath(streams, net, classIdx, posLabel, classNames, cfg)
    % Build per-stream vectors so bootstrap resamples streams, not windows.
    sbcs = cell(numel(streams),1); ybcs = cell(numel(streams),1);
    for i = 1:numel(streams)
        [x, fs] = audioread(streams(i).wavPath);
        if size(x,2) > 1, x = mean(x,2); end
        if fs ~= 16000, x = resample(x,16000,fs); fs = 16000; end
        x = single(max(-1,min(1,x)));

        % frames + stacks
        B = getf(cfg,'features','baseBands',40);
        frameMs = getf(cfg,'features','frameMs',25);
        hopMs   = getf(cfg,'features','hopMs',10);
        L = getf(cfg,'sainath','leftCtx',23); R=getf(cfg,'sainath','rightCtx',8);
        W = L+1+R;

        [Mlog, Tcenters_ms] = fullLogMelMatrix(x, fs, B, frameMs, hopMs);
        c_ms = mean(streams(i).winTimesMs,2);
        idxFrames = round((c_ms - Tcenters_ms(1)) / hopMs) + 1;

        Nw = numel(idxFrames);
        X4 = zeros(B, W, 1, Nw, 'single');
        for j = 1:Nw
            c = idxFrames(j);
            left  = max(1, c - L);
            right = min(size(Mlog,2), c + R);
            block = Mlog(:, left:right);
            if size(block,2) < W
                need = W - size(block,2);
                padLeft  = max(0, L - (c-left));
                padRight = need - padLeft;
                if padLeft > 0,  block = [repmat(Mlog(:,left),  1, padLeft),  block]; end
                if padRight > 0, block = [block, repmat(Mlog(:,right), 1, padRight)]; end
            end
            X4(:,:,1,j) = block(:,1:W);
        end
        mu = mean(X4(:)); sd = std(X4(:)) + eps; X4z = (X4 - mu) / sd;
        S  = predict(net, X4z);
        sbcs{i} = S(:,classIdx);
        ybcs{i} = strcmp(string(streams(i).winLabels(:)), string(posLabel));
    end
end

function [Mlog, t_ms] = fullLogMelMatrix(x, fs, numBands, frameMs, hopMs)
    % Compute log-mel frames for entire stream; return centers in ms (for mapping)
    frameLen    = round(fs * frameMs/1000);
    hopSamp     = max(1, round(fs * hopMs /1000));
    ovl         = max(0, frameLen - hopSamp);
    win         = localHamming(frameLen);

    if exist('melSpectrogram','file') == 2
        try
            M = melSpectrogram(x, fs, 'Window',win, 'OverlapLength',ovl, ...
                               'NumBands', numBands, 'FrequencyRange', [50 min(7000,fs/2*0.999)]);
            Mlog = log10(M + eps);
        catch
            M = melSpectrogram(x, fs, 'Window',win, 'OverlapLength',ovl, 'NumBands', numBands);
            Mlog = log10(M + eps);
        end
    else
        if exist('designAuditoryFilterBank','file') ~= 2
            error('Audio Toolbox function "designAuditoryFilterBank" not found.');
        end
        S = spectrogram(x, win, ovl, numel(win), fs);
        fb = designAuditoryFilterBank(fs, 'NumBands',numBands, 'FFTLength',numel(win), ...
                                      'FrequencyRange', [50 min(7000,fs/2*0.999)]);
        Mlog = log10(fb * abs(S) + eps);
    end

    % time centers in ms
    nFrames = size(Mlog,2);
    t_centers = ((0:nFrames-1) * hopSamp + frameLen/2) / fs;   % seconds
    t_ms = t_centers(:) * 1000;
end

function w = localHamming(N)
    try, w = hamming(N,'periodic'); catch, w = hamming(N); end
end

function cls = tryGetClasses(net)
    try, cls = string(net.Layers(end).Classes); catch, cls = "unknown"; end
end

function v = getf(cfg, group, name, def)
    v = def;
    if ~isstruct(cfg), return; end
    if isfield(cfg, group)
        g = cfg.(group);
        if isfield(g, name) && ~isempty(g.(name)), v = g.(name); end
    end
end

function pos = choosePositiveLabel(cfg, classNames, labels)
    if isfield(cfg,'experiments') && ~isempty(cfg.experiments.forcePosLabel)
        forced = string(cfg.experiments.forcePosLabel);
        if any(classNames == forced), pos = char(forced); return;
        else, error('Forced positiveLabel "%s" not in model classes.', forced);
        end
    end
    labs = categories(labels); counts = zeros(numel(labs),1);
    for i=1:numel(labs), counts(i) = sum(labels==labs{i}); end
    [~, idx] = sort(counts,'descend'); ordered = labs(idx);
    fillers = ["_silence_","_unknown_","_background_noise_","_neg_"];
    for i=1:numel(ordered)
        if ~ismember(string(ordered{i}), fillers), pos = ordered{i}; return; end
    end
    pos = ordered{1};
end
end
