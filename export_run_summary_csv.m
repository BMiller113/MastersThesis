function export_run_summary_csv(resultsRoot)
% export_run_summary_csv
% Consolidate Top-1 Accuracy + AUC (+ FR/FA if available) from existing Results/metrics MAT files.
% Output: <Results>/run_summary.csv
%
% Usage:
%   export_run_summary_csv
%   export_run_summary_csv('C:\...\KeywordSpottingThesis\Results')

if nargin < 1 || isempty(resultsRoot)
    resultsRoot = fullfile(pwd, 'Results');
end
if ~exist(resultsRoot,'dir')
    error('Results folder not found: %s', resultsRoot);
end

metricsFiles = dir(fullfile(resultsRoot, '**', 'metrics', '*.mat'));
if isempty(metricsFiles)
    % fallback: any mat under Results that looks like metrics
    metricsFiles = dir(fullfile(resultsRoot, '**', '*.mat'));
end

rows = {};
for k = 1:numel(metricsFiles)
    p = fullfile(metricsFiles(k).folder, metricsFiles(k).name);
    try
        S = load(p);
    catch
        continue;
    end

    % Identify metrics struct + rocInfo struct
    metrics = struct();
    rocInfo = struct();

    if isfield(S,'metrics') && isstruct(S.metrics), metrics = S.metrics; end
    if isfield(S,'rocInfo') && isstruct(S.rocInfo), rocInfo = S.rocInfo; end
    if isempty(fieldnames(metrics)) && isfield(S,'results') && isstruct(S.results)
        % some branches save results instead
        metrics = S.results;
    end

    % Pull Top-1 accuracy
    top1 = pullScalar(metrics, {'Top1Accuracy','Top1Acc','Accuracy','Acc'});
    if isnan(top1)
        continue; % not a run metrics file
    end

    % Pull FR/FA if available
    FR = pullScalar(metrics, {'FR','FalseReject','FRpercent'});
    FA = pullScalar(metrics, {'FA','FalseAlarm','FApercent'});

    % AUC: prefer stored; otherwise compute from far/frr
    auc = NaN;
    if isfield(metrics,'AUC') && isnumeric(metrics.AUC) && isscalar(metrics.AUC)
        auc = double(metrics.AUC);
    elseif isstruct(rocInfo) && isfield(rocInfo,'AUC') && ~isempty(rocInfo.AUC)
        auc = double(rocInfo.AUC);
    elseif isstruct(rocInfo) && isfield(rocInfo,'far') && isfield(rocInfo,'frr')
        auc = computeAUCfromFarFrr(rocInfo.far, rocInfo.frr);
    end

    % Parse metadata from filename
    [ds, arch, runMode, melMode, B, W] = parseFromName(metricsFiles(k).name);

    rows(end+1,:) = {ds, arch, runMode, melMode, B, W, top1, auc, FR, FA, p}; %#ok<AGROW>
end

if isempty(rows)
    error('No usable metrics MAT files found under %s.', resultsRoot);
end

T = cell2table(rows, 'VariableNames', ...
    {'Dataset','Architecture','RunMode','MelMode','FreqBins','Frames','Top1Accuracy','AUC','FR','FA','SourceMat'});

% Normalize types
T.Dataset      = string(T.Dataset);
T.Architecture = string(T.Architecture);
T.RunMode      = string(T.RunMode);
T.MelMode      = string(T.MelMode);
T.SourceMat    = string(T.SourceMat);

T.FreqBins     = toDoubleOrNaN(T.FreqBins);
T.Frames       = toDoubleOrNaN(T.Frames);
T.Top1Accuracy = toDoubleOrNaN(T.Top1Accuracy);
T.AUC          = toDoubleOrNaN(T.AUC);
T.FR           = toDoubleOrNaN(T.FR);
T.FA           = toDoubleOrNaN(T.FA);

% Sort: dataset then accuracy desc
T = sortrows(T, {'Dataset','Top1Accuracy'}, {'ascend','descend'});

outCsv = fullfile(resultsRoot, 'run_summary.csv');
writetable(T, outCsv);
fprintf('Wrote: %s (%d rows)\n', outCsv, height(T));

end

% ---------------- helpers ----------------

function v = pullScalar(S, names)
v = NaN;
if ~isstruct(S), return; end
for i = 1:numel(names)
    nm = names{i};
    if isfield(S,nm)
        x = S.(nm);
        if isnumeric(x) && isscalar(x)
            v = double(x); return;
        end
    end
end
end

function auc = computeAUCfromFarFrr(far, frr)
far = far(:); frr = frr(:);
tpr = 1 - frr;

[farS, idx] = sort(far);
tprS = tpr(idx);

ok = ~(isnan(farS) | isnan(tprS));
farS = farS(ok); tprS = tprS(ok);

if numel(farS) < 2
    auc = NaN;
else
    auc = trapz(farS, tprS);
end
end

function [ds, arch, runMode, melMode, B, W] = parseFromName(fname)
s = lower(erase(string(fname), ".mat"));

% Dataset
ds = "unknown";
if contains(s,"v1"), ds = "v1"; end
if contains(s,"v2"), ds = "v2"; end

% Architecture
arch = "unknown";
archList = ["one-fstride4","one_fstride4","tpool2","trad-fpool3","trad_fpool3"];
for a = archList
    if contains(s,a)
        arch = replace(a,"_","-");
        break;
    end
end

% Run mode
runMode = "unknown";
if contains(s,"mel-only") || contains(s,"mel_only"), runMode = "mel-only"; end
if contains(s,"none"), runMode = "none"; end

% Mel mode
melMode = "unknown";
melList = ["default","narrow","wide","prop7k","prop8k","linear"];
for mm = melList
    if contains(s,mm), melMode = mm; break; end
end

% Geometry like 40x32
B = NaN; W = NaN;
tok = regexp(s,'(\d+)\s*x\s*(\d+)','tokens','once');
if ~isempty(tok)
    B = str2double(tok{1});
    W = str2double(tok{2});
end
end

function x = toDoubleOrNaN(col)
try
    x = double(col);
catch
    if iscell(col)
        x = nan(size(col));
        for i = 1:numel(col)
            x(i) = str2double(string(col{i}));
        end
    else
        x = str2double(string(col));
    end
end
end
