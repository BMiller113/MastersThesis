function plot_sainath_curves(curveDirOrFiles, varargin)
% Make ROC and/or DET plots from sainath_curve_*.csv
% Usage:
%   plot_sainath_curves(curvesDir)
%   plot_sainath_curves(curvesDir, 'saveDir', outDir, 'addDET', true, 'perModel', true)
%
% Inputs:
%   curveDirOrFiles : folder containing sainath_curve_*.csv OR cellstr of CSV paths
%
% Name-Value:
%   'saveDir'   : where to save figures (defaults to same folder as curves)
%   'addDET'    : also generate DET plots (default true)
%   'perModel'  : save per-model ROC/DET figures (default true)
%   'xlimROC'   : x-limit for ROC (in percent) e.g., [0 5]
%   'ylimROC'   : y-limit for ROC (TPR, 0..1) e.g., [0 1]
%   'visible'   : 'off' for headless (default), 'on' to pop windows

% --------- Args / defaults ---------
p = inputParser;
addRequired(p, 'curveDirOrFiles');
addParameter(p, 'saveDir', '', @ischar);
addParameter(p, 'addDET', true, @(x)islogical(x) || isnumeric(x));
addParameter(p, 'perModel', true, @(x)islogical(x) || isnumeric(x));
addParameter(p, 'xlimROC', [0 5], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'ylimROC', [0 1], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'visible', 'off', @(s)ischar(s) && any(strcmpi(s,{'on','off'})));
parse(p, curveDirOrFiles, varargin{:});
args = p.Results;

% Discover files
files = {};
if ischar(curveDirOrFiles) && exist(curveDirOrFiles,'dir') == 7
    curvesDir = curveDirOrFiles;
    d = dir(fullfile(curvesDir, 'sainath_curve_*.csv'));
    if ~isempty(d)
        files = fullfile({d.folder}, {d.name});
    end
else
    files = curveDirOrFiles;
    if ischar(files), files = {files}; end
    if isempty(files)
        warning('plot_sainath_curves: no files provided.'); 
        return;
    end
    curvesDir = fileparts(files{1});
end

if isempty(files)
    warning('plot_sainath_curves: no curve CSVs found at %s', curveDirOrFiles);
    return;
end

% Save dir
saveDir = args.saveDir;
if isempty(saveDir), saveDir = fullfile(curvesDir, '..', 'plots'); end
if ~exist(saveDir,'dir'), mkdir(saveDir); end

% Load all curves
C = struct('tag',{},'thr',{},'fpr',{},'tpr',{},'frr',{},'fa_h',{},'fr_pct',{},'fa_pct',{});
for i = 1:numel(files)
    T = readtable(files{i});
    need = {'threshold','fpr','tpr','FRpercent','FApercent','FA_per_hr'};
    for k = 1:numel(need)
        if ~ismember(need{k}, T.Properties.VariableNames)
            error('Curve CSV missing column: %s\nFile: %s', need{k}, files{i});
        end
    end
    [~,bn,~] = fileparts(files{i});
    tag = erase(bn, 'sainath_curve_');
    C(i).tag    = tag;
    C(i).thr    = T.threshold(:);
    C(i).fpr    = T.fpr(:);
    C(i).tpr    = T.tpr(:);
    C(i).frr    = 1 - T.tpr(:);
    C(i).fa_h   = T.FA_per_hr(:);
    C(i).fr_pct = T.FRpercent(:);
    C(i).fa_pct = T.FApercent(:);
end

% --------- Overlay ROC (FPR% vs TPR) ---------
figROC = figure('Visible',args.visible); hold on; grid on;
for i = 1:numel(C)
    plot(C(i).fpr*100, C(i).tpr, 'LineWidth', 2);
end
xlabel('False Positive Rate (%)'); ylabel('True Positive Rate');
title('Sainath-style ROC (overlay)');
legend({C.tag}, 'Interpreter','none','Location','SouthEast');
xlim(args.xlimROC); ylim(args.ylimROC);
saveas(figROC, fullfile(saveDir, 'sainath_ROC_overlay.png'));
savefig(figROC, fullfile(saveDir, 'sainath_ROC_overlay.fig'));
close(figROC);

% --------- Overlay DET (optional) ---------
if args.addDET
    figDET = figure('Visible',args.visible); hold on; grid on;
    for i = 1:numel(C)
        x = localNormInv(clamp01(C(i).fpr));
        y = localNormInv(clamp01(C(i).frr));
        plot(x, y, 'LineWidth', 2);
    end
    % Set probit-style ticks (common DET % ticks)
    tickPct = [0.1 0.2 0.5 1 2 5 10 20 40 60];
    tickP   = tickPct/100;
    xticks(localNormInv(tickP)); xticklabels(strcat(string(tickPct),'%'));
    yticks(localNormInv(tickP)); yticklabels(strcat(string(tickPct),'%'));
    xlabel('False Alarm rate'); ylabel('Miss rate (FRR)');
    title('DET (overlay)');
    legend({C.tag}, 'Interpreter','none','Location','SouthWest');
    saveas(figDET, fullfile(saveDir, 'sainath_DET_overlay.png'));
    savefig(figDET, fullfile(saveDir, 'sainath_DET_overlay.fig'));
    close(figDET);
end

% --------- Per-model figures (optional) ---------
if args.perModel
    for i = 1:numel(C)
        % ROC (per model)
        f1 = figure('Visible',args.visible); grid on; hold on;
        plot(C(i).fpr*100, C(i).tpr, 'LineWidth', 2);
        xlabel('False Positive Rate (%)'); ylabel('True Positive Rate');
        title(sprintf('ROC — %s', C(i).tag), 'Interpreter','none');
        xlim(args.xlimROC); ylim(args.ylimROC);
        saveas(f1, fullfile(saveDir, sprintf('sainath_ROC_%s.png', C(i).tag)));
        savefig(f1, fullfile(saveDir, sprintf('sainath_ROC_%s.fig', C(i).tag)));
        close(f1);

        % DET (per model)
        if args.addDET
            f2 = figure('Visible',args.visible); grid on; hold on;
            x = localNormInv(clamp01(C(i).fpr));
            y = localNormInv(clamp01(C(i).frr));
            plot(x, y, 'LineWidth', 2);
            tickPct = [0.1 0.2 0.5 1 2 5 10 20 40 60];
            tickP   = tickPct/100;
            xticks(localNormInv(tickP)); xticklabels(strcat(string(tickPct),'%'));
            yticks(localNormInv(tickP)); yticklabels(strcat(string(tickPct),'%'));
            xlabel('False Alarm rate'); ylabel('Miss rate (FRR)');
            title(sprintf('DET — %s', C(i).tag), 'Interpreter','none');
            saveas(f2, fullfile(saveDir, sprintf('sainath_DET_%s.png', C(i).tag)));
            savefig(f2, fullfile(saveDir, sprintf('sainath_DET_%s.fig', C(i).tag)));
            close(f2);
        end
    end
end

fprintf('Saved plots to: %s\n', saveDir);

% --------- helpers ------------------
function z = localNormInv(p)
    % NormInv without Stats TB: norminv(p) = -sqrt(2)*erfcinv(2p)
    p = clamp01(p);
    z = -sqrt(2) * erfcinv(2*p);
end
function y = clamp01(x)
    % Clamp to open interval (0,1) to avoid +/-Inf in norm inverse.
    eps1 = 1e-12;
    y = min(max(x, eps1), 1-eps1);
end
end
