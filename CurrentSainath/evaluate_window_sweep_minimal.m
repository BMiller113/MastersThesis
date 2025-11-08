function curvePath = evaluate_window_sweep_minimal(modelFile, streamsFile, conditionTag, outCsvPath)
% evaluate_window_sweep_minimal
% Window-level eval, 10 ms decisions, Sainath 14 KWs.
% This version is robust to the "clean = 0 FA/h everywhere" case:
%   - thresholds come from actual scores + an extra 0 at the end
%   - try to clip to FA/h<=2000
%   - if that leaves <2 points, keep the full curve
%   - if that STILL leaves <2, make a 2nd point so plotting works

    if nargin < 3 || isempty(conditionTag)
        conditionTag = 'clean';
    end

    % ----- load model -----
    S = load(modelFile);
    if ~isfield(S,'net'), error('Model file missing "net".'); end
    net = S.net;

    % ----- cfg -----
    cfg = kws_config('sainath14');
    cfg.streaming.hopWinMs = 10;
    hopSec = cfg.streaming.hopWinMs / 1000;
    decisionsPerHour = 3600 / hopSec;   % 360000

    % ----- load streams -----
    L = load(streamsFile);
    if isfield(L,'streams_clean') && strcmpi(conditionTag,'clean')
        streams = L.streams_clean;
    elseif isfield(L,'streams_noisy') && strcmpi(conditionTag,'noisy')
        streams = L.streams_noisy;
    elseif isfield(L,'streams')
        streams = L.streams;
    else
        error('Could not find streams in %s', streamsFile);
    end
    streams = ewm_fix_stream_paths(streams, streamsFile);

    % ----- collect scores/labels -----
    kwList = lower(string(cfg.sainath.targetWords(:)));
    scores_all = [];
    labels_all = [];

    for i = 1:numel(streams)
        wavPath = streams(i).wavPath;
        if ~isfile(wavPath)
            error('Stream %d WAV not found: %s', i, wavPath);
        end

        Sw = slidingWindowScores(net, wavPath, cfg);

        classNames = lower(string(Sw.classes));
        kwMask = ismember(classNames, kwList);
        if ~any(kwMask)
            kwMask = ~startsWith(classNames, "_");
        end
        pKW = max(Sw.scores(:, kwMask), [], 2);

        labs = string(streams(i).winLabels);
        n = min(numel(pKW), numel(labs));
        pKW = pKW(1:n);
        labs = labs(1:n);

        scores_all = [scores_all; pKW(:)]; %#ok<AGROW>
        labels_all = [labels_all; labs(:)]; %#ok<AGROW>
    end

    % tiny deterministic jitter so negatives don't all tie
    Ntot = numel(scores_all);
    if Ntot > 1
        scores_all = scores_all + (1e-7) * ((1:Ntot).' / Ntot);
    end

    % ----- threshold grid from scores + a zero -----
    thr_scores = sort(unique(scores_all), 'descend');
    thr = [thr_scores; 0];   % 0 guarantees "everything fires"  => nonzero FA

    isPos = ismember(lower(labels_all), kwList);
    P = max(1, sum(isPos));
    N = max(1, sum(~isPos));

    Srow = scores_all.'; Yrow = isPos.';
    pred = bsxfun(@ge, Srow, thr);   % K x N

    TP = sum(pred & repmat(Yrow, size(pred,1),1), 2);
    FP = sum(pred & repmat(~Yrow, size(pred,1),1), 2);

    tpr = TP / P;
    frr = 1 - tpr;
    fpr = FP / N;

    FA_per_hr   = fpr * decisionsPerHour;
    FR_fraction = frr;

    % ----- collapse identical FA bins -----
    [FAu, ~, grp] = unique(FA_per_hr, 'stable');
    FRu = accumarray(grp, FR_fraction, [], @mean);
    Thu = accumarray(grp, thr,          [], @mean);

    FA_per_hr   = FAu;
    FR_fraction = FRu;
    thr         = Thu;

    % ----- try to clip to visible range -----
    maxFA = 2000;
    keep = FA_per_hr <= maxFA;
    FA_clip   = FA_per_hr(keep);
    FR_clip   = FR_fraction(keep);
    thr_clip  = thr(keep);

    if numel(FA_clip) >= 2
        % good, use clipped
        FA_per_hr   = FA_clip;
        FR_fraction = FR_clip;
        thr         = thr_clip;
    else
        % clipped got too small -> use full un-clipped
        % (CLEAN case right now)
        % if even full has <2 points, synthesize one
        if numel(FA_per_hr) < 2
            FA_per_hr   = [FA_per_hr; decisionsPerHour];
            FR_fraction = [FR_fraction; FR_fraction(end)];
            thr         = [thr; 0];
        end
    end

    % ----- write CSV -----
    cfg2 = kws_config('sainath14');
    if nargin < 4 || isempty(outCsvPath)
        [~,bn,~] = fileparts(modelFile);
        tag = erase(bn, 'model_');
        outDir = fullfile(cfg2.paths.outputDir, 'sainath', conditionTag, 'curves');
        if ~exist(outDir,'dir'), mkdir(outDir); end
        outCsvPath = fullfile(outDir, sprintf('sainath_curve_events_%s.csv', tag));
    end

    T = table(thr, FA_per_hr, FR_fraction, ...
        'VariableNames', {'threshold','FA_per_hr','FR_fraction'});
    writetable(T, outCsvPath);

    fprintf('evaluate_window_sweep_minimal: %s (%s): pts=%d, windows=%d, pos=%d, neg=%d\n', ...
        outCsvPath, conditionTag, numel(FA_per_hr), numel(scores_all), sum(isPos), sum(~isPos));

    if nargout > 0
        curvePath = outCsvPath;
    end
end

% ---------------------------------------------------------
function streams = ewm_fix_stream_paths(streams, streamsMatPath)
    baseDir = fileparts(streamsMatPath);
    for s = 1:numel(streams)
        wp = streams(s).wavPath;
        if isfile(wp), continue; end

        wp2 = fullfile(baseDir, wp);
        if isfile(wp2)
            streams(s).wavPath = wp2; continue;
        end

        if isfolder(wp)
            guess = fullfile(wp, sprintf('stream_%02d.wav', s));
            if isfile(guess), streams(s).wavPath = guess; continue; end
            guess2 = fullfile(baseDir, wp, sprintf('stream_%02d.wav', s));
            if isfile(guess2), streams(s).wavPath = guess2; continue; end
        end

        guess3 = fullfile(baseDir, sprintf('stream_%02d.wav', s));
        if isfile(guess3), streams(s).wavPath = guess3; end
    end
end
