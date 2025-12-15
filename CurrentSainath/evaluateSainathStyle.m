function res = evaluateSainathStyle(net, streams, cfg)
% evaluateSainathStyle
% Sainath-style streaming evaluation on a pre-built "streams" corpus.
% - Concatenates all window-level scores and labels across streams
% - Threshold-sweeps to compute ROC/DET; maps FPR -> FA/hour from hop
% - Operating point = EER unless cfg.experiments.fixedThreshold is set
% - Optional: FR@FA/hour targets; bootstrap CIs over streams

verbose = false;  % quiet, true for popups

% --- scores/labels/classes ---
[allScores, allLabels, classNames] = collectScoresAndLabels(net, streams, cfg);

% Positive class
posLabel = choosePositiveLabel(cfg, classNames, allLabels);
classIdx = find(strcmp(classNames, posLabel), 1);
if isempty(classIdx), error('Positive label "%s" not found.', posLabel); end

scores = allScores(:, classIdx);                 % Nx1 posterior for pos class
y = logical(strcmp(string(allLabels), string(posLabel)));  % Nx1 logical

% if no pos/neg, give warning to avoid curve collapse
Npos = sum(y); Nneg = sum(~y);
if Npos == 0
    warning('No positives for "%s" — TPR=0, FR%%=100.', posLabel);
end
if Nneg == 0
    error('No negatives in windows — cannot compute FPR/FAh.');
end
Npos = max(1,Npos); Nneg = max(1,Nneg);

% --- Build threshold grid ---
if isempty(scores)
    thrGrid = linspace(0,1,2001).';
else
    lo = max(0, min(scores) - 1e-6);
    hi = min(1, max(scores) + 1e-6);
    thrGrid = linspace(lo, hi, 2001).';
end
% If config provides a grid, only use it if it’s reasonable
if isfield(cfg,'sainath') && isfield(cfg.sainath,'thrGrid') && ~isempty(cfg.sainath.thrGrid)
    if numel(cfg.sainath.thrGrid) >= 50
        thrGrid = cfg.sainath.thrGrid(:);
    else
        warning('cfg.sainath.thrGrid has %d point(s) — ignoring and using auto-grid.', numel(cfg.sainath.thrGrid));
    end
end

% Vectorized sweep
S = scores(:)'; Y = y(:)'; T = thrGrid(:)';
predPos = S >= T.';                 % KxN
TP = sum(predPos &  Y, 2);          % Kx1
FP = sum(predPos & ~Y, 2);
tpr = TP ./ Npos;
fpr = FP ./ Nneg;
frr = 1 - tpr;

% Sort by FPR, make area computation stable
[fpr, idx] = sort(fpr, 'ascend');
tpr = tpr(idx); frr = frr(idx); thrGrid = thrGrid(idx);

FR_percent = frr * 100;
FA_percent = fpr * 100;

hopMs = getfield_def(cfg,'streaming','hopWinMs',100);
decisionsPerHour = 3600 / (hopMs/1000);
FA_per_hr = fpr * decisionsPerHour;

AUC = trapz(fpr, tpr);

% Operating point
if isfield(cfg,'experiments') && ~isempty(cfg.experiments.fixedThreshold)
    thr_used = cfg.experiments.fixedThreshold;
else
    [~, kEER] = min(abs(fpr - (1 - tpr)));
    thr_used = thrGrid(kEER);
end
pred_op = scores >= thr_used;
FR_at_op_percent = (sum(~pred_op & y)  / max(1,sum(y)))  * 100;
FAh_at_op        = (sum( pred_op & ~y) / max(1,sum(~y))) * decisionsPerHour;

% FA/h targets
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

% Optional stream level bootstrap
if isfield(cfg,'sainath') && isfield(cfg.sainath,'bootstrapN') && cfg.sainath.bootstrapN > 0
    B = cfg.sainath.bootstrapN;
    [sbcs, ybcs] = splitByStream(streams, net, classIdx, posLabel, classNames, cfg);
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
else
    CI_AUC = []; CI_FR = []; CI_FAh = [];
end

% Pack
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
    fprintf('AUC=%.4f | EER thr=%.4f | FR@op=%.2f%% | FA/h@op=%.2f\n', AUC, thr_used, FR_at_op_percent, FAh_at_op);
end

% --------------helpers --------------
function [tpr_, fpr_, frr_] = sweepROC(s_, y_, thr_)
    s_ = s_(:)'; y_ = y_(:)'; thr_ = thr_(:)';
    pred_ = s_ >= thr_.';
    P_ = max(1,sum(y_==1)); N_ = max(1,sum(y_==0));
    TP_ = sum(pred_ &  y_, 2); FP_ = sum(pred_ & ~y_, 2);
    tpr_ = TP_ ./ P_; fpr_ = FP_ ./ N_; frr_ = 1 - tpr_;
end

function [scoresAll, labelsAll, classNames] = collectScoresAndLabels(net, streams, cfg)
    scoresAll = []; labelsAll = categorical();
    classNames = tryGetClasses(net);
    for i = 1:numel(streams)
        % --- Require windows here (created in run_sainath_benchmark) ---
        if ~(isfield(streams(i),'winTimesMs') && isfield(streams(i),'winLabels'))
            error('Stream %d has no winTimesMs/winLabels (did you rebuild cache after code change?).', i);
        end

        % Build features from wav + windows
        X4 = featuresFromWavAndWindows(streams(i), cfg);
        S  = predict(net, X4);
        if isempty(scoresAll), scoresAll = S; else, scoresAll = [scoresAll; S]; end %#ok<AGROW>
        labelsAll = [labelsAll; streams(i).winLabels(:)]; %#ok<AGROW>
    end

    % Constrain label cats into model domain when possible
    if isstring(classNames) || iscellstr(classNames)
        extra = setdiff(categories(labelsAll), cellstr(classNames));
        if ~isempty(extra) && any(strcmp(classNames,"_unknown_"))
            labelsAll = mergecats(labelsAll, extra, "_unknown_");
        end
    end
end

function X4 = featuresFromWavAndWindows(s, cfg)
    % Build a 4-D feature tensor from wav + window bounds (ms)
    srTgt = getfield_def(cfg,'features','forceSampleRate',16000);
    [x, fs] = audioread(s.wavPath);
    if size(x,2) > 1, x = mean(x,2); end
    if fs ~= srTgt, x = resample(x, srTgt, fs); fs = srTgt; end

    starts = s.winStartIdx(:); ends = s.winEndIdx(:);
    starts = max(starts,1); ends = min(ends, numel(x)); ends = max(ends, starts);

    % Materialize temp wavs per window
    tmpDir = tempname; mkdir(tmpDir);
    c = onCleanup(@() safeRmDir(tmpDir));
    nW = numel(starts);
    pathsTmp = cell(nW,1);
    for j = 1:nW
        seg = x(starts(j):ends(j));
        mx  = max(abs(seg)); if mx>1, seg = seg./mx; end
        p = fullfile(tmpDir, sprintf('win_%06d.wav', j));
        audiowrite(p, seg, fs);
        pathsTmp{j} = p;
    end
    X4 = extractFeatures(pathsTmp, 'all', 'default', cfg);
end

function pos = choosePositiveLabel(cfg, classNames, labels)
    if isfield(cfg,'experiments') && ~isempty(cfg.experiments.fixedThreshold) && ...
       isfield(cfg,'experiments') && ~isempty(cfg.experiments.forcePosLabel)
        forced = string(cfg.experiments.forcePosLabel);
        if any(classNames == forced), pos = char(forced); return;
        else, error('Forced positiveLabel "%s" not in model classes.', forced);
        end
    end
    % default: most frequent non-filler
    labs = categories(labels); counts = zeros(numel(labs),1);
    for i=1:numel(labs), counts(i) = sum(labels==labs{i}); end
    [~, idx] = sort(counts,'descend'); ordered = labs(idx);
    fillers = ["_silence_","_unknown_","_background_noise_","_neg_"];
    for i=1:numel(ordered)
        if ~ismember(string(ordered{i}), fillers), pos = ordered{i}; return; end
    end
    pos = ordered{1};
end

function cls = tryGetClasses(net)
    try, cls = string(net.Layers(end).Classes); catch, cls = "unknown"; end
end

function v = getfield_def(S, group, name, def)
    v = def;
    if ~isstruct(S), return; end
    if isfield(S, group)
        g = S.(group);
        if isfield(g, name) && ~isempty(g.(name)), v = g.(name); end
    end
end

function safeRmDir(d)
    if exist(d,'dir'), try, rmdir(d,'s'); catch, end, end
end

function [sbcs, ybcs] = splitByStream(streams, net, classIdx, posLabel, classNames, cfg)
    sbcs = cell(numel(streams),1); ybcs = cell(numel(streams),1);
    for i = 1:numel(streams)
        X4 = featuresFromWavAndWindows(streams(i), cfg);
        S  = predict(net, X4);
        sbcs{i} = S(:,classIdx);
        ybcs{i} = strcmp(string(streams(i).winLabels(:)), string(posLabel));
    end
end
end
