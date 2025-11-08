function run_sainath_benchmark(outDirOverride)
% run_sainath_benchmark
% Batch Sainathstyle streaming evaluation over saved models.
%
% What it does:
%   - Loads cfg = kws_config()
%   - Finds all "model_*.mat" in cfg.paths.outputDir (or override)
%   - Builds (or loads) a cached streaming corpus from your test split
%   - Adds sliding windows and labels that match front end
%   - Evaluates each model with evaluateSainathStyle()
%   - Writes results to specified areas
%   - Prints a short final table to the terminal

    % ---- 0) Config & folders ----
    cfg = kws_config();
    if ~isfield(cfg,'sainath'), cfg.sainath = struct(); end
    cfg.sainath.thrGrid = [];  % let evaluator build a dense grid from scores

    if nargin >= 1 && ~isempty(outDirOverride)
        baseOut = outDirOverride;
    else
        baseOut = cfg.paths.outputDir;
    end
    if ~exist(baseOut,'dir')
        error('Output dir does not exist: %s', baseOut);
    end

    sainathRoot = fullfile(baseOut, 'sainath');
    dirSummary  = fullfile(sainathRoot, 'summary');
    dirCurves   = fullfile(sainathRoot, 'curves');
    dirOps      = fullfile(sainathRoot, 'ops');       % per-target operating points
    dirMats     = fullfile(sainathRoot, 'mats');
    dirStreams  = fullfile(sainathRoot, 'streams');
    if ~exist(dirSummary,'dir'), mkdir(dirSummary); end
    if ~exist(dirCurves,'dir'),  mkdir(dirCurves);  end
    if ~exist(dirOps,'dir'),     mkdir(dirOps);     end
    if ~exist(dirMats,'dir'),    mkdir(dirMats);    end
    if ~exist(dirStreams,'dir'), mkdir(dirStreams); end

    % ---- 1) Find models ----
    models = dir(fullfile(baseOut, 'model_*.mat'));
    if isempty(models)
        fprintf('No model_*.mat files found in %s\n', baseOut);
        return;
    end

    % ---- 2) Build/load streaming corpus once ----
    % Capture noisy prints from loadAudioData/makeStreamingCorpus
    try
        evalc('[~, ~, testF, testL] = loadAudioData();');
    catch ME
        error('Failed to load dataset (loadAudioData): %s', ME.message);
    end

    % Derive a window span that matches your feature extractor:
    % spanMs ≈ frameMs + hopMs*(targetFrames-1)
    spanMs = getfield_def(cfg,'features','frameMs',30) + ...
             getfield_def(cfg,'features','hopMs',10) * ...
             (getfield_def(cfg,'features','targetFrames',98) - 1);
    hopMs     = getfield_def(cfg,'streaming','hopWinMs', 100);  % decisions every 100 ms
    tolMs     = getfield_def(cfg,'sainath','labelTolMs', 100);  % small timing slack for event match
    minOvlMs  = getfield_def(cfg,'sainath','minOverlapMs', 20); % require about 20 ms overlap to call a positive

    cacheName = sprintf('streams_v2_span%d_hop%d_tol%d_ovl%d.mat', round(spanMs), hopMs, tolMs, minOvlMs);
    cachePath = fullfile(dirStreams, cacheName);

    if exist(cachePath,'file')
        S = load(cachePath, 'streams');
        streams = S.streams;
    else
        outWavDir = fullfile(dirStreams, 'wav');
        if ~exist(outWavDir,'dir'), mkdir(outWavDir); end
        try
            evalc('streams = makeStreamingCorpus(cfg, testF, testL, outWavDir);');
        catch ME
            error('makeStreamingCorpus failed: %s', ME.message);
        end
        % Add sliding windows + labels that match the model’s receptive field (this is critical, and likely needs refinement)
        streams = addSlidingWindowsToStreams(streams, spanMs, hopMs, tolMs, minOvlMs);
        save(cachePath, 'streams', '-v7.3');
    end
    if isempty(streams)
        error('Streaming corpus is empty.');
    end

    % Sanity check
    evalc('sanityPrint(streams)');

    % ---- 3) Evaluate each model; make summary ----
    SummTag = {}; SummAUC = []; SummFAh = []; SummFR = []; SummThr = [];
    SummPos = {}; SummArch = {};
    SummCI_AUC_lo = []; SummCI_AUC_hi = [];
    SummCI_FR_lo  = []; SummCI_FR_hi  = [];
    SummCI_FAh_lo = []; SummCI_FAh_hi = [];

    for k = 1:numel(models)
        modelFile = fullfile(models(k).folder, models(k).name);
        [~, baseName, ~] = fileparts(modelFile);
        tag  = erase(baseName, 'model_');          % e.g. "mel-only_wide" or "none_default"
        arch = tryGetArch(modelFile, cfg);

        % Load network
        try
            S = load(modelFile, 'net', 'ARCH_TYPE');
        catch ME
            warning('Skipping %s (load failed): %s', models(k).name, ME.message);
            continue;
        end
        if ~isfield(S,'net')
            warning('Skipping %s: no variable "net" found.', models(k).name);
            continue;
        end
        net = S.net;

        % --- Auto-sync feature geometry to input size ---
        try
            inSize = net.Layers(1).InputSize;  % [H W C]
        catch
            warning('Could not read InputSize for %s; assuming [40 98 1].', models(k).name);
            inSize = [40 98 1];
        end
        Hexp = inSize(1);  Wexp = inSize(2);  Cexp = inSize(3);
        if Cexp ~= 1
            warning('Model %s expects %d channels; this eval assumes 1.', models(k).name, Cexp);
        end
        
        %10/6 suggestion !!!
        % Local eval cfg matched to this model (CRITICAL)
        cfgEval = cfg;  % copy base cfg and override only geometry
        cfgEval.features.baseBands    = Hexp;
        cfgEval.features.targetFrames = Wexp;
        % keep your front-end timing unless you know the model was trained
        % with something else; these values make Wexp=98 natural.
        % cfgEval.features.frameMs = 30;
        % cfgEval.features.hopMs   = 10;

        % Evaluate one time with the per-model geometry
        try
            res = evaluateSainathStyle(net, streams, cfgEval);
        catch ME
            warning('Evaluation failed for %s: %s', models(k).name, ME.message);
            continue;
        end

        % Save detailed artifacts
        matOut = fullfile(dirMats,   sprintf('sainath_res_%s.mat', tag));
        csvOut = fullfile(dirCurves, sprintf('sainath_curve_%s.csv', tag));
        try, save(matOut, 'res', '-v7.3'); catch ME, warning('Could not save MAT for %s: %s', tag, ME.message); end
        try
            Tcurve = table( ...
                res.thresholds(:), res.fpr(:), res.tpr(:), ...
                res.FR_percent(:), res.FA_percent(:), res.FA_per_hr(:), ...
                'VariableNames', {'threshold','fpr','tpr','FRpercent','FApercent','FA_per_hr'});
            writetable(Tcurve, csvOut);
        catch ME
            warning('Could not save curve CSV for %s: %s', tag, ME.message);
        end

        % Operating points at FA/h targets (if any)
        if isfield(res,'faTargets') && ~isempty(res.faTargets)
            try
                Tops = table(res.faTargets(:), res.FR_at_targets(:), res.thr_at_targets(:), ...
                    'VariableNames', {'FA_per_hr_target','FRpercent','threshold'});
                writetable(Tops, fullfile(dirOps, sprintf('sainath_ops_%s.csv', tag)));
            catch ME
                warning('Could not save ops CSV for %s: %s', tag, ME.message);
            end
        end

        % Summary row
        SummTag{end+1,1}  = tag;
        SummArch{end+1,1} = arch;
        SummPos{end+1,1}  = res.positiveLabel;
        SummAUC(end+1,1)  = res.AUC;
        SummFAh(end+1,1)  = res.FAh_at_op;
        SummFR(end+1,1)   = res.FR_at_op_percent;
        SummThr(end+1,1)  = res.thr_used;

        if isfield(res,'CI_AUC')
            SummCI_AUC_lo(end+1,1) = res.CI_AUC(1); %#ok<AGROW>
            SummCI_AUC_hi(end+1,1) = res.CI_AUC(2);
            SummCI_FR_lo(end+1,1)  = res.CI_FR(1);
            SummCI_FR_hi(end+1,1)  = res.CI_FR(2);
            SummCI_FAh_lo(end+1,1) = res.CI_FAh(1);
            SummCI_FAh_hi(end+1,1) = res.CI_FAh(2);
        else
            [SummCI_AUC_lo(end+1,1), SummCI_AUC_hi(end+1,1)] = deal(NaN); %#ok<AGROW>
            [SummCI_FR_lo(end+1,1),  SummCI_FR_hi(end+1,1)]  = deal(NaN);
            [SummCI_FAh_lo(end+1,1), SummCI_FAh_hi(end+1,1)] = deal(NaN);
        end
    end

    % ---- 4) Final summary table, print output ----
    Tsum = table(SummTag, SummArch, SummPos, SummAUC, SummFAh, SummFR, SummThr, ...
        SummCI_AUC_lo, SummCI_AUC_hi, SummCI_FR_lo, SummCI_FR_hi, SummCI_FAh_lo, SummCI_FAh_hi, ...
        'VariableNames', {'ModelTag','Arch','PositiveLabel','AUC','FA_per_hour_at_op','FRpercent_at_op','Thr_used', ...
                          'AUC_CI_lo','AUC_CI_hi','FR_CI_lo','FR_CI_hi','FAh_CI_lo','FAh_CI_hi'});
    sumCSV = fullfile(dirSummary, 'sainath_summary.csv');
    if ~isempty(Tsum)
        try, writetable(Tsum, sumCSV); catch, end
        fprintf('\n=== Sainath-style Streaming Summary ===\n');
        disp(Tsum);
        fprintf('Saved:\n  %s\n  %s\n  %s\n  %s\n', sumCSV, dirCurves, dirOps, dirMats);
    else
        fprintf('No successful model evaluations.\n');
    end

    % Optional: auto-plot saved curves
    try
        plot_sainath_curves(dirCurves, ...
            'saveDir', fullfile(sainathRoot,'plots'), ...
            'addDET', true, ...                 % toggle DET
            'perModel', true, ...               % per-model figures too
            'xlimROC', [0 5], ...               % zoom small-FPR region (in %)
            'ylimROC', [0 1], ...
            'visible', 'off');                  % set 'on' for UI
    catch
    end
end

% ---------------- helpers ----------------
function v = getfield_def(cfg, group, name, def)
    v = def;
    if ~isstruct(cfg), return; end
    if isfield(cfg, group)
        g = cfg.(group);
        if isfield(g, name) && ~isempty(g.(name)), v = g.(name); end
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
        arch = cfg.model.arch;
    else
        arch = 'unknown';
    end
end

function streams = addSlidingWindowsToStreams(streams, spanMs, hopMs, tolMs, minOvlMs)
% Build per-stream sliding windows and label them from the events table.
% - Window times are [startMs endMs] with hop hopMs and span spanMs.
% - A window is positive for label L if its time-interval overlaps an event
%   with label L by at least minOvlMs.
    for s = 1:numel(streams)
        p = streams(s).wavPath;
        if ~exist(p,'file'), error('Stream WAV missing: %s', p); end
        info = audioinfo(p);
        durMs = round(info.Duration * 1000);

        % Window grid (inclusive start, clamp end)
        if durMs < spanMs, nW = 1; else, nW = floor((durMs - spanMs)/hopMs) + 1; end
        starts = (0:(nW-1))' * hopMs;
        ends   = starts + spanMs;
        ends(ends > durMs) = durMs;

        % Default label: _neg_
        winLab = repmat("_neg_", nW, 1);

        % If there are events, assign positives by max overlap
        E = streams(s).events;
        if ~isempty(E)
            evStart = round(1000 * E.onset_s);
            evEnd   = round(1000 * E.offset_s);
            evLab   = string(lower(E.label));

            for e = 1:numel(evStart)
                a0 = evStart(e); a1 = evEnd(e);
                ovl = max(0, min(ends,a1) - max(starts,a0));  % ms overlap
                idx = find(ovl >= minOvlMs);
                if ~isempty(idx)
                    replace = ovl(idx) > 0;
                    winLab(idx(replace)) = evLab(e);
                end
            end
        end

        streams(s).winTimesMs = [starts ends];
        streams(s).winLabels  = categorical(winLab);
        % (also cache sample indices, optional)
        fs = info.SampleRate;
        streams(s).winStartIdx = max(1, round((starts/1000) * fs));
        streams(s).winEndIdx   = max(streams(s).winStartIdx, round((ends/1000) * fs));
    end
end

function sanityPrint(streams)
% Silent, but useful for debugging
    try
        labs = []; for i=1:numel(streams), labs = [labs; streams(i).winLabels(:)]; end
        C = categories(labs);
        fprintf('Windows total: %d\n', numel(labs));
        for i=1:numel(C)
            fprintf('  %-12s : %6d\n', C{i}, sum(labs==C{i}));
        end
    catch
    end
end
