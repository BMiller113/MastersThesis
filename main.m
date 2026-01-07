clear; clc;

cfg = kws_config();

% Choose which dataset to run
datasetsToRun = {'v1'};   % {'v1'} {'v2'} or {'v1','v2'}

% Enforce pooled-only modes so nothing tries to use speaker metadata
cfg.experiments.includeModes = intersect(cfg.experiments.includeModes, {'none','mel-only'}, 'stable');
cfg.experiments.gendersToRun = {'all'};

% Figure visibility
if isfield(cfg,'runtime') && isfield(cfg.runtime,'figureVisibility') && ~isempty(cfg.runtime.figureVisibility)
    set(0,'DefaultFigureVisible', cfg.runtime.figureVisibility);
else
    set(0,'DefaultFigureVisible','on');
end

ensureDir(cfg.paths.outputDir);
ensureDir(cfg.paths.modelDir);
ensureDir(cfg.paths.metricsDir);
ensureDir(cfg.paths.cacheDir);

genderModes = cfg.experiments.includeModes;
melModesAll = cfg.experiments.melModes;

overlayCurves = {};
overlayLabels = {};

ARCH = cfg.model.arch;

for dsi = 1:numel(datasetsToRun)
    cfg.dataset.version = lower(string(datasetsToRun{dsi}));

    for g = 1:numel(genderModes)
        mode = lower(string(genderModes{g}));

        switch mode
            case "none"
                useMelFilter  = false;
                localMelModes = {'default'};

            case "mel-only"
                useMelFilter  = true;
                localMelModes = melModesAll;

            otherwise
                warning('Skipping unsupported pooled-only mode: %s', mode);
                continue;
        end

        for m = 1:numel(localMelModes)
            melMode = lower(string(localMelModes{m}));
            if ~useMelFilter && melMode ~= "default", continue; end

            fprintf('\n=== %s | MEL=%s | GROUP=ALL | DS=%s ===\n', ...
                upper(mode), upper(melMode), upper(cfg.dataset.version));

            % ------------------ Load file lists for this dataset ------------------
            try
                [trainFiles, trainLabels, testFiles, testLabels] = loadAudioData(cfg);
            catch ME
                warning('Data load failed for %s/%s/all/%s: %s', ...
                    mode, melMode, cfg.dataset.version, ME.message);
                continue;
            end

            fprintf('[%s] Train=%d Test=%d\n', upper(cfg.dataset.version), numel(trainFiles), numel(testFiles));

            % ------------------ Extract features ------------------
            try
                if useMelFilter
                    [XTrain, vTr] = extractFeatures(trainFiles, 'all', char(melMode), cfg);
                    [XTest,  vTe] = extractFeatures(testFiles,  'all', char(melMode), cfg);
                else
                    [XTrain, vTr] = extractFeatures(trainFiles, 'all', 'default', cfg);
                    [XTest,  vTe] = extractFeatures(testFiles,  'all', 'default', cfg);
                end
            catch ME
                warning('Feature extraction failed (%s/%s/%s): %s', mode, melMode, cfg.dataset.version, ME.message);
                continue;
            end

            YTrain = trainLabels(vTr);
            YTest  = testLabels(vTe);

            if numel(categories(YTrain)) < 2 || numel(categories(YTest)) < 2
                warning('Not enough classes after filtering/validity. Skipping.');
                continue;
            end

            % ------------------ Geometry ------------------
            freqBins   = size(XTrain,1);
            timeFrames = size(XTrain,2);
            frameMs    = getf(cfg,'features','frameMs',25);
            hopMs      = getf(cfg,'features','hopMs',10);
            timeSpanMs = frameMs + (timeFrames - 1)*hopMs;
            fprintf('CNN input: %d×%d×1 (~%.0f ms span)\n', freqBins, timeFrames, timeSpanMs);

            % ------------------ Build / Train / Cache model ------------------
            layers = defineCNNArchitecture(numel(categories(YTrain)), ARCH, freqBins, timeFrames);

            tag = sprintf('%s_%s_all', mode, melMode);
            runKey = makeRunKey(cfg, ARCH, tag, freqBins, timeFrames);

            try
                [net, modelPath] = getOrTrainModel(tag, XTrain, YTrain, layers, cfg); %#ok<ASGLU>
            catch
                % If you don't have getOrTrainModel in this branch, fall back to training directly
                net = trainCNN(XTrain, YTrain, layers, cfg);
                modelPath = '';
            end

            % ------------------ Evaluate ------------------
            try
                [metrics, rocInfo] = evaluateModel( ...
                    net, XTest, YTest, false, cfg.experiments.forcePosLabel, cfg.experiments.fixedThreshold, cfg);
            catch ME
                warning('Evaluation failed (%s): %s', runKey, ME.message);
                continue;
            end

            % ------------------ Save per-run ROC (pop + disk) ------------------
            if isfield(cfg,'plots') && isfield(cfg.plots,'roc') && cfg.plots.roc.popUpAtEnd
                try
                    plotAndSaveROC(rocInfo, runKey, cfg);
                catch ME
                    warning('plotAndSaveROC failed: %s', ME.message);
                end
            end

            % ------------------ Collect for overlay ------------------
            if isstruct(rocInfo) && isfield(rocInfo,'far') && isfield(rocInfo,'frr') && ~isempty(rocInfo.far)
                overlayCurves{end+1} = rocInfo; %#ok<SAGROW>
                overlayLabels{end+1} = sprintf('%s | %s | %s', ...
                    upper(mode), upper(melMode), upper(cfg.dataset.version)); %#ok<SAGROW>
            end

            % ------------------ Export run artifacts if helper exists ------------------
            if exist('exportRunArtifacts','file') == 2
                try
                    cost = [];
                    if exist('estimateCnnCost','file') == 2
                        cost = estimateCnnCost(layers, [freqBins timeFrames 1]);
                    end
                    exportRunArtifacts(cfg, runKey, metrics, rocInfo, cost, modelPath);
                catch ME
                    warning('exportRunArtifacts failed: %s', ME.message);
                end
            else
                % Minimal save if you don't have exportRunArtifacts in this branch
                outDir = cfg.paths.outputDir;
                save(fullfile(outDir, ['results_' runKey '.mat']), 'metrics', 'rocInfo', 'cfg');
            end

            fprintf('Finished: %s\n', runKey);
        end
    end
end

% ------------------ Final overlay ------------------
if isfield(cfg,'plots') && isfield(cfg.plots,'overlay') && cfg.plots.overlay.enable && ~isempty(overlayCurves)
    overlayKey = sprintf('%s__%s__overlay', lower(cfg.dataset.version), lower(string(cfg.model.arch)));
    overlayKey = regexprep(overlayKey,'[^a-zA-Z0-9_]+','_');
    try
        plotOverlayAndSaveROC(overlayCurves, overlayLabels, overlayKey, cfg);
    catch ME
        warning('Overlay plotting failed: %s', ME.message);
    end
end

fprintf('\nDone.\n');

% ============================ helpers ==============================
function v = getf(cfg, section, name, defaultV)
    v = defaultV;
    if isfield(cfg, section)
        S = cfg.(section);
        if isfield(S, name) && ~isempty(S.(name)), v = S.(name); end
    end
end

function ensureDir(p)
    if ~exist(p,'dir'), mkdir(p); end
end

function runKey = makeRunKey(cfg, arch, tag, B, W)
    ds = lower(string(cfg.dataset.version));
    arch = lower(string(arch));
    tag = lower(string(tag));
    runKey = sprintf('%s__%s__%s__%dx%d', ds, arch, tag, B, W);
    runKey = regexprep(runKey,'[^a-zA-Z0-9_]+','_');
end
% ===================================================================
