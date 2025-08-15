% main.m  —  Keyword Search with Gender Control (config = kws_config)
% 8/15/25

%Load config
cfg = kws_config();

% Apply runtime UI preference (Training progress popups or no)
if isfield(cfg.runtime,'figureVisibility')
    set(0,'DefaultFigureVisible', cfg.runtime.figureVisibility);
end
if isfield(cfg.runtime,'suppressWarnings') && cfg.runtime.suppressWarnings
    warning('off','extractFeatures:filefail');
    warning('off','backtrace');
end

MAKE_PLOTS      = cfg.runtime.makePlots;
FORCE_POS_LABEL = cfg.experiments.forcePosLabel;
FIXED_THRESHOLD = cfg.experiments.fixedThreshold;
ARCH_TYPE       = cfg.model.arch;

% Load gender map
gmFile = cfg.paths.genderMapFile;
if exist(gmFile,'file')
    S = load(gmFile, 'genderMap');
    genderMap = S.genderMap;
else
    error(['%s not found.\n' ...
           'Create it once with:\n' ...
           '   genderMap = build_speaker_gender_map(cfg.paths.datasetRoot);\n' ...
           '   save(cfg.paths.genderMapFile, ''genderMap'');'], gmFile);
end


% Experiment grid (from config)
genderModes = cfg.experiments.includeModes;                 % {'none','filter','filter+mel'}
melModesAll = cfg.experiments.melModes;                     % {'default','narrow','wide','prop7k','prop8k'}
runGenders  = cfg.experiments.gendersToRun;                 % {'all','male','female'}


% Main main
for g = 1:numel(genderModes)
    genderMode = genderModes{g};
    switch genderMode
        case 'none'
            genderList   = intersect({'all'}, runGenders, 'stable');
            useMelFilter = false;
            melModes     = {'default'};
        case 'filter'
            genderList   = intersect({'male','female'}, runGenders, 'stable');
            useMelFilter = false;
            melModes     = {'default'};
        case 'filter+mel'
            genderList   = intersect({'male','female'}, runGenders, 'stable');
            useMelFilter = true;
            melModes     = melModesAll;
        otherwise
            warning('Unknown genderMode: %s (skipping)', genderMode);
            continue;
    end

    for m = 1:numel(melModes)
        melMode = melModes{m};
        if ~useMelFilter && ~strcmpi(melMode,'default'), continue; end

        for gi = 1:numel(genderList)
            filterGender = genderList{gi};  % 'all' | 'male' | 'female'

            fprintf('\n=== MODE: %s | MEL: %s | GROUP: %s ===\n', ...
                upper(genderMode), upper(melMode), upper(filterGender));

            % 1) Load data ONLY (scope errors to loading)
            try
                [XTrain, YTrain, XTest, YTest] = loadGenderSplitData( ...
                    genderMap, filterGender, useMelFilter, melMode);
            catch ME
                warning('Data load failed for %s/%s/%s: %s', genderMode, melMode, filterGender, ME.message);
                continue;
            end

            % 2) Sanity checks (prevent shape/count bugs)-
            assert(size(XTrain,4) == numel(YTrain), ...
                'XTrain items (%d) != YTrain labels (%d)', size(XTrain,4), numel(YTrain));
            assert(size(XTest,4)  == numel(YTest), ...
                'XTest items (%d) != YTest labels (%d)', size(XTest,4), numel(YTest));

            assert(ndims(XTrain)==4 && ndims(XTest)==4, 'Features must be 4-D arrays.');
            assert(size(XTrain,3)==1 && size(XTest,3)==1, 'Expected 1 channel in features (size(:, :, 1, N)).');

            if numel(categories(YTrain)) < 2 || numel(categories(YTest)) < 2
                warning('Not enough classes to train/evaluate (train=%d, test=%d). Skipping.', ...
                    numel(categories(YTrain)), numel(categories(YTest)));
                continue;
            end

            % 3) Define & train (dynamic input size)
            freqBins  = size(XTrain,1);
            timeSteps = size(XTrain,2);

            layers = defineCNNArchitecture( ...
                numel(categories(YTrain)), ARCH_TYPE, freqBins, timeSteps);

            net = trainCNN(XTrain, YTrain, layers, cfg);

            % 4) Evaluate
            [accuracy, FR, FA, rocInfo, dbg] = evaluateModel( ...
                net, XTest, YTest, MAKE_PLOTS, FORCE_POS_LABEL, FIXED_THRESHOLD);

            % 5) Per-class FR/FA & macro stats (optional)
            perClassTbl = [];  % reset each condition to avoid carryover
            MacroFR = NaN; MacroFA = NaN; MacroAUC = NaN;
            if exist('evaluateAllClasses.m','file')
                try
                    [perClassTbl, macro] = evaluateAllClasses(net, XTest, YTest, false);
                    MacroFR  = macro.FR;
                    MacroFA  = macro.FA;
                    MacroAUC = macro.AUC;
                catch ME
                    warning('Per-class evaluation failed');
                end
            end

            % 6) Persist artifacts
            tag = sprintf('%s_%s', genderMode, melMode);
            if ~strcmpi(filterGender, 'all')
                tag = sprintf('%s_%s', tag, filterGender);
            end

            outDir = cfg.paths.outputDir;
            if ~exist(outDir,'dir'), mkdir(outDir); end

            save(fullfile(outDir, ['model_' tag '.mat']), 'net', 'ARCH_TYPE');

            results = struct( ...
                'GenderMode',    genderMode, ...
                'MelMode',       melMode, ...
                'FilterGender',  filterGender, ...
                'Accuracy',      accuracy, ...
                'FR',            FR, ...
                'FA',            FA, ...
                'MacroFR',       MacroFR, ...
                'MacroFA',       MacroFA, ...
                'MacroAUC',      MacroAUC, ...
                'PositiveLabel', rocInfo.positiveLabel, ...
                'ThresholdUsed', rocInfo.thrUsed, ...
                'Timestamp',     datetime());

            save(fullfile(outDir, ['results_' tag '.mat']), 'results', 'rocInfo');

            if ~isempty(perClassTbl) && istable(perClassTbl)
                writetable(perClassTbl, fullfile(outDir, ['results_by_class_' tag '.csv']));
            end

            % Console summary
            fprintf('Acc: %.2f%% | FR: %.2f%% | FA: %.2f%% | POS="%s" | thr=%.4f\n', ...
                accuracy, FR, FA, results.PositiveLabel, results.ThresholdUsed);
            if ~isnan(MacroFR)
                fprintf('Macro (classes): FR=%.2f%% | FA=%.2f%% | AUC=%.3f\n', MacroFR, MacroFA, MacroAUC);
            end
        end
    end
end

% Summarize all runs to CSVs (all + male + female)
try
    summarizeResults(true);   % export CSVs (single arg)
catch ME
    warning('summarizeResults failed');
end

% Re-enable figure visibility at the end (optional)
set(0,'DefaultFigureVisible','on');
