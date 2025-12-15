function curvePath = evaluate_window_sweep_minimal(modelFile, streamsFile, conditionTag, outCsvPath)
% Window FR vs FA/hour in Sainath spirit, 10 ms hop.
%
% Key choices:
%   tp_mode:
%       'event'        -> TP if max p(true_kw) over the full GT span hits threshold (default)
%       'event_center' -> TP if max p(true_kw) within ±posCenterMs of the GT center hits threshold
%       'window'       -> legacy: TP per positive window
%
%   fa_mode:
%       'transitions'  -> count FA as threshold-crossing bursts on negative timeline
%                         with deadTimeMs refractory (recommended; Sainath-like)
%       'window'       -> legacy per-window FP -> FA/h = FPR * decisionsPerHour
%
% Optional smoothing:
%   enable_smooth = true -> 100–200 ms movmax on per-window KW posteriors (reduces spiky FAs)
%
% CSV columns: threshold, FA_per_hr, FR_fraction

    if nargin < 3 || isempty(conditionTag), conditionTag = 'clean'; end

    % ===== load model =====
    S = load(modelFile);
    if ~isfield(S,'net'), error('Model file missing "net".'); end
    net = S.net;

    % ===== cfg / constants =====
    cfg = kws_config('sainath14');
    hopMs  = 10;     hopSec = hopMs / 1000;   % 10 ms decisions
    decPerHr = 3600 / hopSec;                % 360000

    % ---- Sainath-ish switches ----
    tp_mode        = 'event';       % event-based TPs
    posCenterMs    = 150;           % for 'event_center' (unused here)
    fa_mode        = 'transitions'; % transitions + dead-time
    deadTimeMs     = 1000;          % merge FA bursts more aggressively
    enable_smooth  = true;          % smooth scores before thresholding
    smoothMs       = 200;           % ~200 ms smoothing

    % ===== load streams =====
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
    streams = local_fix_stream_paths(streams, streamsFile);

    % ===== target list =====
    kwList = lower(string(cfg.sainath.targetWords(:)));

    % ===== accumulators =====
    posScores_all = [];   % per-event scores
    negScores_all = [];   % per-window maxKW scores for negatives
    negIdx_all    = [];   % [startRow endRow] per-stream into negScores_all
    streamHours   = 0;    % total evaluated hours
    P = 0; N = 0;

    % ===== iterate streams =====
    for i = 1:numel(streams)
        wavPath = streams(i).wavPath;
        if ~isfile(wavPath), error('Missing WAV: %s', wavPath); end

        Sw = slidingWindowScores(net, wavPath, cfg);      % .scores, .classes
        classNames = string(lower(Sw.classes));
        kwMask = ismember(classNames, kwList);
        if ~any(kwMask)    % fallback: anything not starting with '_'
            kwMask = ~startsWith(classNames, "_");
        end

        scoresKW = Sw.scores(:, kwMask);   % [Nwin x K]
        labs     = string(lower(streams(i).winLabels));
        nwin     = min(size(scoresKW,1), numel(labs));
        scoresKW = scoresKW(1:nwin, :);
        labs     = labs(1:nwin);

        % optional smoothing on KW posteriors
        if enable_smooth
            win = max(1, round(smoothMs / hopMs));
            win = 2*floor(win/2)+1; % make odd
            scoresKW = movmax(scoresKW, [floor(win/2) floor(win/2)], 1, 'Endpoints','shrink');
        end

        % window times
        if isfield(streams(i),'winTimes') && numel(streams(i).winTimes) >= nwin
            tsec = streams(i).winTimes(1:nwin); %#ok<NASGU>
        else
            tsec = ((0:nwin-1).' + 0.5) * hopSec; %#ok<NASGU>
        end

        % Negatives: windows labeled _neg_; score = max over KWs
        isNegWin = (labs == "_neg_");
        if any(isNegWin)
            p_any = max(scoresKW(isNegWin, :), [], 2);
            ng0   = numel(negScores_all) + 1;
            negScores_all = [negScores_all; p_any]; %#ok<AGROW>
            ng1   = numel(negScores_all);
            negIdx_all    = [negIdx_all; ng0, ng1]; %#ok<AGROW>
            N = N + numel(p_any);

            streamHours = streamHours + numel(isNegWin) * hopSec / 3600;
        end

        % Positives -> event grouping
        isPosWin = ismember(labs, kwList);
        switch tp_mode
            case 'window'
                if any(isPosWin)
                    posLabs = labs(isPosWin);
                    kwNames = string(classNames(kwMask));
                    [tf, colIdx] = ismember(posLabs, kwNames);
                    rows = find(isPosWin);
                    rows = rows(tf);
                    cols = colIdx(tf);
                    p_true = scoresKW(sub2ind(size(scoresKW), rows, cols));
                    posScores_all = [posScores_all; p_true]; %#ok<AGROW>
                    P = P + numel(p_true);
                end

            case {'event','event_center'}
                if any(isPosWin)
                    runs = find_runs(isPosWin);
                    kwNames = string(classNames(kwMask));
                    for r = 1:size(runs,1)
                        a = runs(r,1); b = runs(r,2);
                        kw_here = labs(a);
                        [tf, colIdx] = ismember(kw_here, kwNames);
                        if ~tf, continue; end

                        if strcmp(tp_mode,'event_center')
                            c  = round((a + b)/2);
                            rad = max(1, round((posCenterMs / hopMs)));
                            a2 = max(1, c - rad);  b2 = min(nwin, c + rad);
                            p_true_span = scoresKW(a2:b2, colIdx);
                        else
                            p_true_span = scoresKW(a:b, colIdx);
                        end

                        posScores_all = [posScores_all; max(p_true_span)]; %#ok<AGROW>
                        P = P + 1;
                    end
                end

            otherwise
                error('Unknown tp_mode: %s', tp_mode);
        end
    end

    % ===== sanity + diagnostics =====
    if P == 0, warning('No positives found; seeding one zero.'); posScores_all = 0; P = 1; end
    if N == 0, warning('No negatives found; seeding one zero.'); negScores_all = 0; N = 1; end

    fprintf('Label mix: P=%d, N=%d (P%%=%.3f)\n', P, N, P/(P+N));
    qs = [0.00 0.10 0.25 0.50 0.75 0.90 0.99];
    sp = quantile(posScores_all, qs);
    sn = quantile(max(negScores_all,0), qs);
    fprintf('Score percentiles (pos vs neg(maxKW)):\n');
    for j = 1:numel(qs)
        fprintf('  q=%4.2f  pos=%.4f   neg=%.4f\n', qs(j), sp(j), sn(j));
    end

    % small jitter to break ties
    posScores_all = add_jitter(posScores_all);
    negScores_all = add_jitter(negScores_all);

    % ===== thresholds =====
    thrCore = unique([posScores_all; negScores_all], 'stable');
    thr = [inf; thrCore; 0; -1e-12];

    FA_per_hr   = zeros(numel(thr),1);
    FR_fraction = zeros(numel(thr),1);

    % Pre-sort positives for fast TP counts
    posSorted = sort(posScores_all, 'descend');   % length P

    switch fa_mode
        case 'window'
            negSorted = sort(negScores_all, 'descend'); % length N
            ip = 1; ineg = 1; TP = 0; FP = 0;
            for t = 1:numel(thr)
                th = thr(t);
                if isfinite(th)
                    while ip   <= P &&   posSorted(ip) >= th,   TP = TP + 1;   ip   = ip + 1; end
                    while ineg <= N &&   negSorted(ineg) >= th, FP = FP + 1; ineg = ineg + 1; end
                end
                FR_fraction(t) = 1 - (TP / P);
                fpr = FP / N;
                FA_per_hr(t)   = fpr * decPerHr;
            end

        case 'transitions'
            deadSteps = max(1, round(deadTimeMs / hopMs));
            for t = 1:numel(thr)
                th = thr(t);
                TP = sum(posSorted >= th);
                FR_fraction(t) = 1 - (TP / P);

                FA_count = 0;
                for s = 1:size(negIdx_all,1)
                    g0 = negIdx_all(s,1); g1 = negIdx_all(s,2);
                    if g1 >= g0
                        above = negScores_all(g0:g1) >= th;
                        FA_count = FA_count + count_bursts_with_deadtime(above, deadSteps);
                    end
                end
                FA_per_hr(t) = FA_count / max(streamHours, eps);
            end

        otherwise
            error('Unknown fa_mode: %s', fa_mode);
    end

    % collapse identical FA bins (stable)
    [FAu, ~, grp] = unique(FA_per_hr, 'stable');
    FRu = accumarray(grp, FR_fraction, [], @mean);
    Thu = accumarray(grp, thr,          [], @mean);

    FA_per_hr   = FAu;
    FR_fraction = FRu;
    thr         = Thu;
    fprintf('unique_FA_bins (post-collapse=runlen): %d\n', numel(FA_per_hr));

    % ===== write CSV =====
    cfg2 = kws_config('sainath14');
    if nargin < 4 || isempty(outCsvPath)
        [~,bn,~] = fileparts(modelFile);
        tag = erase(bn, 'model_');
        outDir = fullfile(cfg2.paths.outputDir, 'sainath', conditionTag, 'curves');
        if ~exist(outDir,'dir'), mkdir(outDir); end

        if strcmpi(tp_mode,'window')
            suffix = 'windows';
        elseif strcmpi(tp_mode,'event_center')
            suffix = 'eventCenter';
        else
            suffix = 'events';
        end

        if strcmpi(fa_mode,'transitions')
            suffix = [suffix '_transFA100ms'];
        else
            suffix = [suffix '_winFA'];
        end

        outCsvPath = fullfile(outDir, sprintf('sainath_curve_%s_%s.csv', suffix, tag));
    end

    T = table(thr, FA_per_hr, FR_fraction, ...
        'VariableNames', {'threshold','FA_per_hr','FR_fraction'});
    writetable(T, outCsvPath);

    fprintf('evaluate_window_sweep_minimal: %s (%s): pts=%d, P=%d, N=%d\n', ...
        outCsvPath, lower(conditionTag), numel(FA_per_hr), P, N);

    if nargout > 0
        curvePath = outCsvPath;
    end
end

% ==== helpers ====
function streams = local_fix_stream_paths(streams, streamsMatPath)
    baseDir = fileparts(streamsMatPath);
    for s = 1:numel(streams)
        wp = streams(s).wavPath;
        if isfile(wp), continue; end
        wp2 = fullfile(baseDir, wp);
        if isfile(wp2), streams(s).wavPath = wp2; continue; end
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

function runs = find_runs(mask)
    mask = mask(:) > 0;
    if ~any(mask), runs = zeros(0,2); return; end
    d = diff([false; mask; false]);
    s = find(d == 1);
    e = find(d == -1) - 1;
    runs = [s e];
end

function x = add_jitter(x)
    if numel(x) > 1
        x = x + (1e-9) * ((1:numel(x)).' / numel(x));
    end
end

function n = count_bursts_with_deadtime(above, deadSteps)
    n = 0;
    i = 1;
    L = numel(above);
    while i <= L
        if above(i)
            n = n + 1;
            j = i + 1;
            while j <= L && above(j), j = j + 1; end
            j = min(L + 1, j + deadSteps);
            i = j;
        else
            i = i + 1;
        end
    end
end
