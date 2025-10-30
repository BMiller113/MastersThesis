function [curvePath, curveAll, AUC, FAh_at_op, FR_at_op_percent, thr_used, meanFR1FA] = evaluate_events_and_sweep(net, streams, cfg, outCsvPath)
% EVALUATE_EVENTS_AND_SWEEP (robust broadcasting version)
% Simple Sainath-style WINDOW-LEVEL sweep:
%  - trigger score = max posterior over keyword classes per decision window
%  - sweep a single global threshold over [0..1]
%  - compute FR (fraction) vs FA/hour
%
% Outputs:
%   curveAll.threshold   [Kx1]
%   curveAll.FA_per_hr   [Kx1]
%   curveAll.FR_fraction [Kx1] (0..1)
%
% This version fixes "Arrays have incompatible sizes" by using bsxfun for
% threshold expansion, and adds guards for NaNs/dupes/monotonic FR.

% ---------- output path ----------
if nargin < 4 || isempty(outCsvPath)
    outDir = '';
    try
        if isfield(cfg,'paths') && isfield(cfg.paths,'outputDir')
            outDir = cfg.paths.outputDir;
        end
    catch, end
    if isempty(outDir), outDir = fullfile(pwd,'Results','sainath','_auto'); end
    if ~exist(outDir,'dir'), mkdir(outDir); end
    outCsvPath = fullfile(outDir, sprintf('sainath_curve_events_%s.csv', datestr(now,'yyyymmdd_HHMMSSFFF')));
end

% ---------- decision cadence ----------
hopSec = getHopSec(cfg);                    % seconds between decisions
decisionsPerHour = 3600 / hopSec;

% ---------- target words (keywords) ----------
kwList = getKWList(cfg);                    % lowercased string array

% ---------- gather scores + labels over all streams ----------
scores_all = [];     % pKW per window
labels_all = [];     % 1 if window contains ANY keyword, else 0

for i = 1:numel(streams)
    % Window labels
    if isfield(streams(i),'winLabels') && ~isempty(streams(i).winLabels)
        wlab = string(streams(i).winLabels(:));
    else
        error('Stream %d missing winLabels. Rebuild streams cache.', i);
    end

    % Scores per window at same cadence
    S = slidingWindowScores(net, streams(i).wavPath, cfg);  % S.scores: [Nw x C]
    if isempty(S) || ~isfield(S,'scores') || isempty(S.scores)
        error('No scores from slidingWindowScores for stream %d.', i);
    end

    % Map to keyword trigger pKW
    classNames = getNetClasses(net);
    kwMask = ismember(lower(classNames), kwList);
    if ~any(kwMask)
        % fallback: treat all non-underscore classes as KW
        kwMask = ~startsWith(classNames,"_");
    end
    pKW = max(S.scores(:, kwMask), [], 2);    % [Nw x 1]

    % Length match (truncate to shorter)
    n = min(numel(pKW), numel(wlab));
    if n <= 1
        % nothing useful from this stream
        continue;
    end
    pKW = pKW(1:n);
    wlab = wlab(1:n);

    % Window = positive if label is any keyword
    isKW = ismember(lower(wlab), kwList);

    scores_all = [scores_all; pKW(:)];             %#ok<AGROW>
    labels_all = [labels_all; double(isKW(:))];    %#ok<AGROW>
end

if isempty(scores_all)
    error('No scores collected — verify streams and models.');
end

% ---------- clean scores/labels ----------
mask = isfinite(scores_all);
scores = scores_all(mask);
y      = labels_all(mask) > 0;   % logical
if isempty(scores)
    error('All scores are non-finite.');
end

% ---------- sweep thresholds ----------
P = max(1, sum(y==1));           % #positives
N = max(1, sum(y==0));           % #negatives

% auto grid from score support
lo = max(0, min(scores) - 1e-6);
hi = min(1, max(scores) + 1e-6);
thr = linspace(lo, hi, 4001).';

% Vectorized sweep (robust broadcasting)
Srow = scores(:)';    % 1 x N
Tcol = thr(:);        % K x 1
% predPos(k,n) := (scores(n) >= thr(k))
predPos = bsxfun(@ge, Srow, Tcol);                 % K x N

% Split labels into row for broadcasting too
Yrow = y(:)';                                      % 1 x N
TP = sum( predPos &  repmat(Yrow, size(predPos,1), 1), 2);
FP = sum( predPos & ~repmat(Yrow, size(predPos,1), 1), 2);

tpr = TP ./ P;
fpr = FP ./ N;
frr = 1 - tpr;

% Sort by FPR for stability
[fpr, ord] = sort(fpr, 'ascend');
tpr = tpr(ord); frr = frr(ord); thr = thr(ord);

% Convert to FA/hour
FA_per_hr   = fpr * decisionsPerHour;
FR_fraction = frr;

% ---- dedupe identical FA bins; keep best (min FR) at each FA ----
[FAu, ~, grp] = unique(FA_per_hr, 'stable');
FRu = accumarray(grp, FR_fraction, [], @min);
Thu = accumarray(grp, thr,          [], @mean);    % avg threshold within bin

FA_per_hr   = FAu(:);
FR_fraction = FRu(:);
thr         = Thu(:);

% ---- enforce monotone FR (non-increasing with FA) to avoid little rises ---
FR_fraction = flip(cummin(flip(FR_fraction)));

% ---- ensure at least 2 distinct points for plotting/CSV ---
if numel(FA_per_hr) < 2 || numel(unique(FA_per_hr)) < 2
    % Add a tiny epsilon bin to avoid flatline plotting/CSV consumers
    FA_per_hr   = [FA_per_hr;   FA_per_hr(end) + max(1e-6, 0.001)];
    FR_fraction = [FR_fraction; FR_fraction(end)];
    thr         = [thr; thr(end)];
end

% AUC (TPR vs FPR, not used in FR/FA plot but useful)
AUC = trapz(FA_per_hr/decisionsPerHour, 1-FR_fraction); % integrate TPR over FPR

% Operating point @ 1 FA/h
targetFAh = 1.0;
[FAh_at_op, FR_at_op_percent, thr_used] = op_at_target(FA_per_hr, FR_fraction, thr, targetFAh);
meanFR1FA = FR_at_op_percent;

% ---------- write CSV ----------
T = table(thr(:), FA_per_hr(:), FR_fraction(:), ...
          'VariableNames', {'threshold','FA_per_hr','FR_fraction'});
curvePath = outCsvPath;
writetable(T, curvePath);

% ---------- pack ----------
curveAll = struct('threshold',thr(:), 'FA_per_hr',FA_per_hr(:), 'FR_fraction',FR_fraction(:));
end

% ==================== helpers ====================
function hop = getHopSec(cfg)
hop = [];
if isfield(cfg,'streaming') && isfield(cfg.streaming,'hopWinMs') && ~isempty(cfg.streaming.hopWinMs)
    hop = cfg.streaming.hopWinMs / 1000;
end
if isempty(hop) || ~isfinite(hop) || hop<=0
    if isfield(cfg,'features') && isfield(cfg.features,'hopMs') && ~isempty(cfg.features.hopMs)
        hop = cfg.features.hopMs / 1000;
    else
        hop = 0.01; % 10 ms default
    end
end
end

function kws = getKWList(cfg)
kws = strings(0,1);
if isfield(cfg,'sainath') && isfield(cfg.sainath,'targetWords') && ~isempty(cfg.sainath.targetWords)
    kws = lower(string(cfg.sainath.targetWords(:)'));
elseif isfield(cfg,'warden') && isfield(cfg.warden,'targetWords') && ~isempty(cfg.warden.targetWords)
    kws = lower(string(cfg.warden.targetWords(:)'));
end
kws = kws(:);
end

function classes = getNetClasses(net)
classes = strings(0,1);
try, classes = string(net.Layers(end).Classes); return; catch, end
try, classes = string(net.Classes); return; catch, end
try, classes = string(categories(net.ClassNames)); return; catch, end
end

function [faOp, frPctOp, thrOp] = op_at_target(FAh, FR, thr, target)
faOp = NaN; frPctOp = NaN; thrOp = NaN;
if numel(FAh) < 1, return; end
[FAh, ix] = sort(FAh(:),'ascend'); FR = FR(ix); thr = thr(ix);
if target <= FAh(1), faOp=FAh(1); frPctOp=100*FR(1); thrOp=thr(1); return; end
if target >= FAh(end), faOp=FAh(end); frPctOp=100*FR(end); thrOp=thr(end); return; end
k = find(FAh <= target, 1, 'last');
x0=FAh(k); x1=FAh(k+1); y0=FR(k); y1=FR(k+1);
w = (target - x0) / max(eps, x1-x0);
frInterp = y0 + w*(y1-y0);
thrSel   = thr(k + (w>0.5));
faOp     = target;
frPctOp  = 100*frInterp;
thrOp    = thrSel;
end
