function run_sainath_benchmark(outDirOverride, cfgProfile)
% Robust Sainath-style runner that auto-detects evaluator signatures
% and cleans (FAh, FR, thr) to equal length & sane shape.

% ---------- config ----------
if nargin < 2 || isempty(cfgProfile), cfgProfile = 'sainath14'; end
cfg = kws_config(cfgProfile);

if nargin >= 1 && ~isempty(outDirOverride)
    baseOut = outDirOverride;
else
    assert(isfield(cfg,'paths') && isfield(cfg.paths,'outputDir'), ...
        'cfg.paths.outputDir missing. Set it in kws_config.');
    baseOut = cfg.paths.outputDir;
end
if ~exist(baseOut,'dir'), mkdir(baseOut); end

% Sainath-ish geometry
cfg.features.baseBands     = 40;
cfg.features.frameMs       = 25;
cfg.features.hopMs         = 10;
cfg.features.targetFrames  = 32;
cfg.sainath.leftCtx        = 23;
cfg.sainath.rightCtx       = 8;

% Threshold grid (if needed by evaluator)
if ~isfield(cfg,'sainath') || ~isfield(cfg.sainath,'thrGrid') || isempty(cfg.sainath.thrGrid)
    cfg.sainath.thrGrid = linspace(0,1,801);
end

sainathRoot = fullfile(baseOut, 'sainath');
conds = {'clean','noisy'};

% ---------- models ----------
models = dir(fullfile(baseOut, 'model_*.mat'));
if isempty(models)
    fprintf('No model_*.mat files found in %s\n', baseOut);
    return;
end

% ---------- dataset ----------
try
    evalc('[~, ~, testF, testL] = loadAudioData();');
catch ME
    error('Failed to load dataset (loadAudioData): %s', ME.message);
end

% ---------- run both conditions ----------
for ci = 1:numel(conds)
    cond = conds{ci};
    fprintf('\n=== %s condition ===\n', upper(cond));

    % Dirs
    dirSummary  = fullfile(sainathRoot, cond, 'summary');     ensureDir(dirSummary);
    dirCurves   = fullfile(sainathRoot, cond, 'curves');      ensureDir(dirCurves);
    dirMats     = fullfile(sainathRoot, cond, 'mats');        ensureDir(dirMats);
    dirFRFA     = fullfile(sainathRoot, cond, 'plots_FR_FA'); ensureDir(dirFRFA);
    dirStreams  = fullfile(sainathRoot, cond, 'streams');     ensureDir(dirStreams);

    % Condition-specific noise
    cfg2 = cfg;
    if strcmp(cond,'clean')
        cfg2.streaming.bgGain     = 0.0;
        cfg2.streaming.noiseSNRdB = [];
    else
        if ~isfield(cfg2,'streaming'), cfg2.streaming = struct(); end
        cfg2.streaming.noiseSNRdB = getf(cfg2,'streaming','noiseSNRdB',10);
        cfg2.streaming.bgGain     = getf(cfg2,'streaming','bgGain',0.3);
    end
    % decisions at 10 ms
    cfg2.streaming.hopWinMs = cfg2.features.hopMs;

    % Build or load streams cache
    spanMs = cfg2.features.frameMs + cfg2.features.hopMs*(cfg2.sainath.leftCtx+cfg2.sainath.rightCtx);
    hopMs  = cfg2.streaming.hopWinMs;
    tolMs  = getf(cfg2,'sainath','labelTolMs',100);
    cacheName = sprintf('streams_SAINATH_%s_span%d_hop%d_tol%d.mat', cond, round(spanMs), hopMs, tolMs);
    cachePath = fullfile(dirStreams, cacheName);

    if exist(cachePath,'file')
        S = load(cachePath, 'streams'); streams = S.streams;
    else
        wavDir = fullfile(dirStreams, 'wav'); ensureDir(wavDir);
        streams = makeStreamingCorpus(cfg2, testF, testL, wavDir);
        % sanity
        totGT = 0;
        for ii = 1:numel(streams)
            if isfield(streams(ii),'events') && ~isempty(streams(ii).events)
                totGT = totGT + height(streams(ii).events);
            end
        end
        if totGT == 0
            error('Streaming GT events empty—check makeStreamingCorpus.');
        end
        save(cachePath, 'streams', '-v7.3');
    end

    % ------------ evaluator selection ------------
    use_windows_simple = (exist('evaluate_windows_simple','file')==2);
    have_sainath_simple= (exist('evaluate_sainath_simple','file')==2);
    fprintf('Evaluator: %s\n', tern(use_windows_simple,'evaluate_windows_simple', ...
                               tern(have_sainath_simple,'evaluate_sainath_simple','<none found>')));
    if ~use_windows_simple && ~have_sainath_simple
        error('No evaluator found. Ensure evaluate_windows_simple.m or evaluate_sainath_simple.m is on path.');
    end

    % Summary accumulators
    SummTag={}; SummArch={}; SummAUC=[]; SummFAh=[]; SummFR=[]; SummThr=[];

    for k = 1:numel(models)
        modelFile = fullfile(models(k).folder, models(k).name);
        [~, baseName, ~] = fileparts(modelFile);
        tag  = erase(baseName, 'model_');
        arch = tryGetArch(modelFile, cfg2);

        % Load net
        try
            S = load(modelFile, 'net', 'ARCH_TYPE');
        catch ME
            warning('Skipping %s (load failed): %s', models(k).name, ME.message);
            continue;
        end
        if ~isfield(S,'net'), warning('Skipping %s: no "net".', models(k).name); continue; end
        net = S.net;

        % Require 40x32x1
        try inSz = net.Layers(1).InputSize; catch, inSz = [NaN NaN NaN]; end
        if any(inSz(1:2) ~= [40 32]) || inSz(3) ~= 1
            warning('Skipping %s: model input %dx%dx%d != 40x32x1.', models(k).name, inSz(1),inSz(2),inSz(3));
            continue;
        end

        % -------- run evaluator (robust to signature) --------
        try
            if use_windows_simple
                [FAh, FR, thr, AUC] = evaluate_windows_simple(net, streams, cfg2);
            else
                try
                    [FAh, FR, thr, AUC] = evaluate_sainath_simple(net, streams, cfg2);
                catch
                    [FAh, FR, thr, AUC] = evaluate_sainath_simple(net, streams, cfg2, cfg2.sainath.thrGrid);
                end
            end
        catch ME
            warning('Eval failed %s: %s', models(k).name, ME.message);
            continue;
        end

        % -------- sanitize curve to avoid table length errors --------
        [FAh, FR, thr, ok, msg] = clean_curve(FAh, FR, thr);
        if ~ok
            warning('Skipping %s curve: %s', models(k).name, msg);
            continue;
        end

        % Write CSV for plotter
        csvOut = fullfile(dirCurves, sprintf('sainath_curve_events_%s.csv', tag));
        T = table(FAh(:), FR(:), thr(:), 'VariableNames', {'FA_per_hr','FR_fraction','threshold'});
        try, writetable(T, csvOut); catch, end

        % Save brief MAT summary
        matOut = fullfile(dirMats, sprintf('sainath_res_events_%s.mat', tag));
        res = struct('curve',struct('FA_per_hr',FAh,'FR_fraction',FR,'threshold',thr), ...
                     'positiveLabel','ALL','AUC',AUC);
        try, save(matOut, 'res', '-v7.3'); catch, end

        % Summary row (use FR at closest FAh=1 if present)
        [~, j1] = min(abs(FAh - 1));
        SummTag{end+1,1}=tag;
        SummArch{end+1,1}=arch;
        SummAUC(end+1,1)=AUC;
        SummFAh(end+1,1)=FAh(j1);
        SummFR(end+1,1)=FR(j1)*100;
        SummThr(end+1,1)=thr(j1);
    end

    % Summary CSV
    Tsum = table(SummTag, SummArch, SummAUC, SummFAh, SummFR, SummThr, ...
                 'VariableNames', {'ModelTag','Arch','AUC','FAh_at_~1','FRpercent_at_~1FAh','Thr_used'});
    sumCSV = fullfile(dirSummary, 'sainath_summary.csv');
    if ~isempty(Tsum)
        try, writetable(Tsum, sumCSV); catch, end
        fprintf('Saved:\n  %s\n  %s\n  %s\n', sumCSV, dirCurves, dirMats);
    else
        fprintf('%s: No successful model evaluations.\n', upper(cond));
    end

    % Plots
    try
        plot_sainath_fr_fa(dirCurves, 'saveDir', dirFRFA, 'perModel', true, 'visible', 'off');
    catch ME
        warning('FR–FA/h plotting failed: %s', ME.message);
    end
end
end

% ---------- local helpers ----------
function ensureDir(d), if ~exist(d,'dir'), mkdir(d); end, end
function v = getf(s, group, name, defaultV)
    v = defaultV;
    if isfield(s,group)
        t = s.(group);
        if isfield(t,name) && ~isempty(t.(name)), v = t.(name); end
    end
end
function arch = tryGetArch(modelFile, cfg)
    arch = '';
    try
        S = load(modelFile,'ARCH_TYPE');
        if isfield(S,'ARCH_TYPE') && ~isempty(S.ARCH_TYPE)
            arch = S.ARCH_TYPE; return;
        end
    catch, end
    if isstruct(cfg) && isfield(cfg,'model') && isfield(cfg.model,'arch')
        arch = cfg.model.arch; else, arch = 'unknown';
    end
end
function s = tern(c,a,b), if c, s=a; else, s=b; end, end

function [FAh, FR, thr, ok, msg] = clean_curve(FAh, FR, thr)
% Align lengths; drop non-finite; sort by FAh; de-dup FA bins; enforce
% non-increasing FR (reverse cummin). Return column vectors.
    ok = false; msg = '';
    if isempty(FAh) || isempty(FR) || isempty(thr)
        msg = 'empty curve arrays'; return;
    end
    FAh = FAh(:); FR = FR(:); thr = thr(:);
    n = min([numel(FAh), numel(FR), numel(thr)]);
    FAh = FAh(1:n); FR = FR(1:n); thr = thr(1:n);

    good = isfinite(FAh) & isfinite(FR) & isfinite(thr) & FAh >= 0 & FR >= 0 & FR <= 1;
    FAh = FAh(good); FR = FR(good); thr = thr(good);
    if numel(FAh) < 2, msg = 'too few points after cleaning'; return; end

    [FAh, idx] = sort(FAh, 'ascend'); FR = FR(idx); thr = thr(idx);

    % de-dup identical FAh bins (keep min FR; pick mean thr)
    [FAu, ~, grp] = unique(FAh, 'stable');
    FRu  = accumarray(grp, FR, [], @min);
    thru = accumarray(grp, thr, [], @mean);

    % enforce ROC-like shape (FR non-increasing as FAh grows)
    FRu  = flip(cummin(flip(FRu)));

    FAh = FAu(:); FR = FRu(:); thr = thru(:);
    ok = numel(FAh) >= 2;
    if ~ok, msg = 'insufficient points after de-dup'; end
end
