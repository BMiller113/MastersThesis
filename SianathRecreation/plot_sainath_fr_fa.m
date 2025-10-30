function plot_sainath_fr_fa(curveDirOrFiles, varargin)
% plot_sainath_fr_fa: Sainath-style FR (fraction) vs FA/hour overlay + per-model plots
% Requires CSV columns: FA_per_hr, FR_fraction  (threshold is optional/ignored here)
%
% Usage examples:
%   plot_sainath_fr_fa('Results\sainath\clean\curves');
%   plot_sainath_fr_fa({'curves\sainath_curve_events_mel-only_default.csv', ...
%                       'curves\sainath_curve_events_none_default.csv'}, ...
%                      'saveDir','Results\sainath\clean\plots_FR_FA', ...
%                      'visible','off', 'xCap',5, 'roiBox',true);

% ---------- Options ----------
p = inputParser;
addRequired(p, 'curveDirOrFiles');
addParameter(p, 'saveDir', '', @ischar);
addParameter(p, 'perModel', true, @(x)islogical(x)||isnumeric(x));
addParameter(p, 'visible', 'off', @(s)ischar(s) && any(strcmpi(s,{'on','off'})));
addParameter(p, 'xCap', 5, @(x)isnumeric(x)&&isscalar(x)&&x>0);       % cap x-axis (FA/h) like the paper (default 0..5)
addParameter(p, 'yCap', 0.8, @(x)isnumeric(x)&&isscalar(x)&&x>0);     % cap y-axis (FR fraction)
addParameter(p, 'roiBox', true, @(x)islogical(x)||isnumeric(x));      % draw dashed ROI [0,2]x[0,0.14]
addParameter(p, 'faTargets', [0.1 0.5 1 2 5], @isnumeric);            % vertical reference lines
addParameter(p, 'useLogX', false, @(x)islogical(x)||isnumeric(x));    % optional log x-axis (NOT in paper)
addParameter(p, 'dedupeMethod', 'minFR', @(s)any(strcmpi(s,{'minFR','meanFR'}))); % on duplicate FA bins
parse(p, curveDirOrFiles, varargin{:});
args = p.Results;

% ---------- Discover files ----------
if ischar(curveDirOrFiles) && isfolder(curveDirOrFiles)
    curvesDir = curveDirOrFiles;
    d = dir(fullfile(curvesDir, 'sainath_curve_events_*.csv'));
    files = fullfile({d.folder},{d.name});
else
    files = curveDirOrFiles;
    if ischar(files), files = {files}; end
    if isempty(files)
        warning('plot_sainath_fr_fa: no input files.');
        return;
    end
    curvesDir = fileparts(files{1});
end

if isempty(files)
    warning('plot_sainath_fr_fa: no CSVs matched.');
    return;
end

% ---------- Save dir ----------
saveDir = args.saveDir;
if isempty(saveDir), saveDir = fullfile(curvesDir, '..', 'plots_FR_FA'); end
if ~exist(saveDir,'dir'), mkdir(saveDir); end

% ---------- Load & clean ----------
C = struct('tag',{},'fa',{},'fr',{});
kept = 0; skipped = 0; reasons = {};
for i = 1:numel(files)
    try
        T = readtable(files{i});
    catch ME
        skipped = skipped + 1; reasons{end+1} = sprintf('read %s: %s', files{i}, ME.message); %#ok<AGROW>
        continue;
    end
    need = {'FA_per_hr','FR_fraction'};
    if ~all(ismember(need, T.Properties.VariableNames))
        skipped = skipped + 1; reasons{end+1} = sprintf('missing cols %s', files{i}); %#ok<AGROW>
        continue;
    end
    fa = T.FA_per_hr(:);
    fr = T.FR_fraction(:);

    % Filter to sane values
    good = isfinite(fa) & isfinite(fr) & fa >= 0 & fr >= 0 & fr <= 1;
    fa = fa(good); fr = fr(good);

    % Need at least 2 distinct points
    if numel(fa) < 2 || numel(unique(fa)) < 2
        skipped = skipped + 1; reasons{end+1} = sprintf('few/distinct FA points %s', files{i}); %#ok<AGROW>
        continue;
    end

    % Sort by FA ascending
    [fa, idx] = sort(fa);
    fr = fr(idx);

    % De-dupe repeated FA bins
    [faU, ~, grp] = unique(fa, 'stable');
    switch lower(args.dedupeMethod)
        case 'minfr'
            frU = accumarray(grp, fr, [], @min);
        otherwise % 'meanFR'
            frU = accumarray(grp, fr, [], @mean);
    end

    % Enforce Sainath ROC shape: FR should be non-increasing as FA grows.
    % Do a reverse cumulative minimum to remove little rises from noisy thresholds.
    frMono = flip(cummin(flip(frU)));

    % Tag from filename
    [~,bn,~] = fileparts(files{i});
    tag = erase(bn,'sainath_curve_events_');

    C(end+1).tag = tag; %#ok<AGROW>
    C(end).fa    = faU;
    C(end).fr    = frMono;
    kept = kept + 1;
end

if kept == 0
    firstReason = 'n/a'; if ~isempty(reasons), firstReason = reasons{1}; end
    warning('plot_sainath_fr_fa: nothing to plot. skipped=%d (e.g., %s)', skipped, firstReason);
    return;
end

% ---------- Axis limits (data-driven & safe) ----------
allFA = cat(1, C.fa);
allFR = cat(1, C.fr);

% Cap to requested xRange but keep a little headroom if everything is identical
xmin = max(0, min(allFA));
xmaxData = max(allFA);
xmax = min(max(args.xCap, eps_expand(xmin, xmaxData)), max(xmin + 1, args.xCap)); % at least 1 wide

ymin = 0;
ymaxData = max(allFR);
ymax = min(max(args.yCap, eps_expand(ymin, ymaxData)), 1); % cap at 1

% ---------- Overlay plot ----------
fig = figure('Visible',args.visible); hold on; grid on;
for i = 1:numel(C)
    % Step-style like DET/ROC sweeps
    [faS, frS] = stairs_like(C(i).fa, C(i).fr);
    plot(faS, frS, 'LineWidth', 2);
end
xlabel('False Alarms per hour');
ylabel('False Rejects');
title('FR vs FA/hour (Sainath-style)');
set_axes_x(fig, args.useLogX);
xlim([xmin xmax]); ylim([ymin ymax]);
legend({C.tag}, 'Interpreter','none','Location','NorthEast');

% ROI dashed box [0,2]x[0,0.14]
if args.roiBox
    plot([0 2 2 0 0],[0 0 0.14 0.14 0], 'k--', 'HandleVisibility','off');
end

% Vertical guides at FA targets (draw only if in-range)
for v = args.faTargets(:).'
    if v >= xmin && v <= xmax
        xline(v,'k:','HandleVisibility','off');
    end
end

% Save overlay
saveas(fig, fullfile(saveDir, 'sainath_FR_vs_FAh_overlay.png'));
savefig(fig, fullfile(saveDir, 'sainath_FR_vs_FAh_overlay.fig'));
close(fig);

% ---------- Per-model panels ----------
if args.perModel
    for i = 1:numel(C)
        f = figure('Visible',args.visible); hold on; grid on;
        [faS, frS] = stairs_like(C(i).fa, C(i).fr);
        plot(faS, frS, 'LineWidth', 2);
        if args.roiBox
            plot([0 2 2 0 0],[0 0 0.14 0.14 0], 'k--', 'HandleVisibility','off');
        end
        for v = args.faTargets(:).'
            if v >= xmin && v <= xmax
                xline(v,'k:','HandleVisibility','off');
            end
        end
        xlabel('False Alarms per hour'); ylabel('False Rejects');
        title(sprintf('FR vs FA/h — %s', C(i).tag), 'Interpreter','none');
        set_axes_x(f, args.useLogX);
        xlim([xmin xmax]); ylim([ymin ymax]);
        saveas(f, fullfile(saveDir, sprintf('sainath_FR_vs_FAh_%s.png', C(i).tag)));
        savefig(f, fullfile(saveDir, sprintf('sainath_FR_vs_FAh_%s.fig', C(i).tag)));
        close(f);
    end
end

fprintf('FR–FA/h: kept=%d, skipped=%d. Saved to %s\n', kept, skipped, saveDir);

% ---------- helpers ----------
function val = eps_expand(lo, hi)
    % Ensures hi > lo for axis limits; adds a tiny cushion if needed
    if ~isfinite(lo) || ~isfinite(hi)
        val = 1;
        return;
    end
    if hi <= lo
        val = lo + max(1, abs(lo))*1e-6;
    else
        val = hi;
    end
end

function [xs, ys] = stairs_like(x, y)
    % Build a step curve without using 'stairs' so we can still sort/clean upstream.
    % Assumes x is strictly increasing and y is the monotone FR.
    x = x(:); y = y(:);
    % start step at first x
    xs = [x.'; x.'];           % duplicate each x
    ys = [y.'; [y(1:end-1); y(end)].']; % hold previous y till the next x
    xs = xs(:).'; ys = ys(:).';
end

function set_axes_x(~, useLog)
    ax = gca;
    if useLog
        set(ax, 'XScale','log');
    else
        set(ax, 'XScale','linear');
    end
end
end
