% ============================== main.m ==============================
% Keyword Search (Pre-Warden) — trains/evaluates using the original
% gender-split pipeline only. No Warden code is referenced or called.
% This version FORCES ROC figure pop-ups and saves PNG+FIG per run.

% 1) Load config and hard-disable any Warden usage (even if cfg has it)
cfg = kws_config();
if isfield(cfg,'warden'), cfg.warden.enable = false; end

% ---- FORCE FIGURES/ROC PLOTS ON (overrides cfg) ----
cfg.runtime.figureVisibility = 'on';
cfg.runtime.makePlots        = true;

% 2) Apply runtime UI preferences (now 'on' so windows pop up)
if isfield(cfg.runtime,'figureVisibility')
    set(0,'DefaultFigureVisible', cfg.runtime.figureVisibility);
else
    set(0,'DefaultFigureVisible','on');
end
if isfield(cfg.runtime,'suppressWarnings') && cfg.runtime.suppressWarnings
    warning('off','extractFeatures:filefail');
    warning('off','backtrace');
end

MAKE_PLOTS      = true; % <-- force ROC plotting
FORCE_POS_LABEL = cfg.experiments.forcePosLabel;
FIXED_THRESHOLD = cfg.experiments.fixedThreshold;
ARCH_TYPE       = cfg.model.arch;

% 3) Load speaker→gender map (same as before)
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

% 4) Experiment grid (NO WARDEN MODES)
genderModes = cfg.experiments.includeModes;                 % {'none','filter','filter+mel','mel-only'}
melModesAll = cfg.experiments.melModes;                     % {'default','narrow','wide','prop7k','prop8k'}
runGenders  = cfg.experiments.gendersToRun;                 % {'all','male','female'}

% 5) Ensure output directory
outDir = cfg.paths.outputDir;
if ~exist(outDir,'dir'), mkdir(outDir); end

% 6) Main loops (modes × gender × melMode)
for g = 1:numel(genderModes)
    genderMode = genderModes{g};
    switch genderMode
        case 'none'
            genderList   = intersect({'all'}, runGenders, 'stable');
            useMelFilter = false;
            baseMelModes = {'default'};
        case 'filter'
            genderList   = intersect({'male','female'}, runGenders, 'stable');
            useMelFilter = false;
            baseMelModes = {'default'};
        case 'filter+mel'
            genderList   = intersect({'male','female'}, runGenders, 'stable');
            useMelFilter = true;
            baseMelModes = melModesAll;
        case 'mel-only'
            genderList   = intersect({'all'}, runGenders, 'stable');   % no split
            useMelFilter = true;
            baseMelModes = melModesAll;
        otherwise
            warning('Unknown genderMode: %s (skipping)', genderMode);
            continue;
    end

    for gi = 1:numel(genderList)
        filterGender = genderList{gi};  % 'all' | 'male' | 'female'

        % Allow optional "linear" bank for female only, if enabled
        localMelModes = baseMelModes;
        if useMelFilter ...
                && strcmpi(filterGender,'female') ...
                && isfield(cfg.experiments,'enableLinearForFemale') ...
                && cfg.experiments.enableLinearForFemale
            if ~ismember('linear', localMelModes)
                localMelModes = [localMelModes, {'linear'}];
            end
        end

        for m = 1:numel(localMelModes)
            melMode = localMelModes{m};
            if ~useMelFilter && ~strcmpi(melMode,'default'), continue; end

            fprintf('\n=== MODE: %s | MEL: %s | GROUP: %s ===\n', ...
                upper(genderMode), upper(melMode), upper(filterGender));

            % -------- Data load (pre-Warden path only) --------
            try
                [XTrain, YTrain, XTest, YTest] = buildFeaturesForMode( ...
                    genderMap, filterGender, useMelFilter, melMode, cfg);
            catch ME
                warning('Data load failed for %s/%s/%s: %s', ...
                    genderMode, melMode, filterGender, ME.message);
                continue;
            end

            % -------- Sanity + geometry log --------
            assert(size(XTrain,4) == numel(YTrain), ...
                'XTrain items (%d) != YTrain labels (%d)', size(XTrain,4), numel(YTrain));
            assert(size(XTest,4)  == numel(YTest), ...
                'XTest items (%d) != YTest labels (%d)', size(XTest,4), numel(YTest));
            assert(ndims(XTrain)==4 && ndims(XTest)==4, 'Features must be 4-D arrays.');
            assert(size(XTrain,3)==1 && size(XTest,3)==1, 'Expected 1 channel (H x W x 1 x N).');

            freqBins   = size(XTrain,1);
            timeFrames = size(XTrain,2);
            frameMs    = getf(cfg,'features','frameMs',25);
            hopMs      = getf(cfg,'features','hopMs',10);
            timeSpanMs = frameMs + (timeFrames - 1)*hopMs;
            fprintf('CNN input: %d×%d×1 (~%.0f ms span)\n', freqBins, timeFrames, timeSpanMs);

            if numel(categories(YTrain)) < 2 || numel(categories(YTest)) < 2
                warning('Not enough classes to train/eval (train=%d, test=%d) — skipping.', ...
                    numel(categories(YTrain)), numel(categories(YTest)));
                continue;
            end

            % -------- Define network from geometry --------
            layers = defineCNNArchitecture( ...
                numel(categories(YTrain)), ARCH_TYPE, freqBins, timeFrames);

            % -------- Train (uses GPU if available; see trainCNN.m) --------
            net = trainCNN(XTrain, YTrain, layers, cfg);

            % -------- Evaluate --------
            [accuracy, FR, FA, rocInfo, dbg] = evaluateModel( ...
                net, XTest, YTest, true, FORCE_POS_LABEL, FIXED_THRESHOLD); %#ok<ASGLU>
            %                     ^^^^ MAKE_PLOTS forced true here

            % -------- Persist artifacts --------
            tag = sprintf('%s_%s', genderMode, melMode);
            if ~strcmpi(filterGender, 'all')
                tag = sprintf('%s_%s', tag, filterGender);
            end

            save(fullfile(outDir, ['model_' tag '.mat']), 'net', 'ARCH_TYPE');

            % Optional per-class breakdown
            perClassTbl = [];
            MacroFR = NaN; MacroFA = NaN; MacroAUC = NaN;
            if exist('evaluateAllClasses.m','file')
                try
                    [perClassTbl, macro] = evaluateAllClasses(net, XTest, YTest, false);
                    MacroFR  = macro.FR;  MacroFA  = macro.FA;  MacroAUC = macro.AUC;
                catch
                    warning('Per-class evaluation failed');
                end
            end

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
                perClassForCSV = perClassTbl;
                vn = perClassForCSV.Properties.VariableNames;
                if ismember('FR', vn)
                    perClassForCSV.FRpercent = perClassForCSV.FR;
                    perClassForCSV = removevars(perClassForCSV, 'FR');
                end
                if ismember('FA', vn)
                    perClassForCSV.FApercent = perClassForCSV.FA;
                    perClassForCSV = removevars(perClassForCSV, 'FA');
                end
                writetable(perClassForCSV, fullfile(outDir, ['results_by_class_' tag '.csv']));
            end

            fprintf('Acc: %.2f%% | FR: %.2f%% | FA: %.2f%% | POS="%s" | thr=%.4f\n', ...
                accuracy, FR, FA, results.PositiveLabel, results.ThresholdUsed);
            if ~isnan(MacroFR)
                fprintf('Macro (classes): FR=%.2f%% | FA=%.2f%% | AUC=%.3f\n', MacroFR, MacroFA, MacroAUC);
            end

            % -------- POP-UP + SAVE ROC (even if evaluateModel didn't) --------
            try
                maybe_plot_and_save_roc(rocInfo, tag, outDir);
            catch ME
                warning('ROC plotting helper failed for %s: %s', tag, ME.message);
            end
        end
    end
end

% 7) Summarize to CSVs/plots (same as before)
origDir = pwd;
try
    if ~isempty(outDir) && exist(outDir,'dir')
        cd(outDir);
    end
    summarizeResults(true, true);   % force plots during summary as well
catch ME
    warning('summarizeResults failed: %s', ME.message);
end
cd(origDir);

% Re-ensure figure visibility at the end
set(0,'DefaultFigureVisible','on');

% ============================ helpers ==============================
function v = getf(s, group, name, defaultV)
    v = defaultV;
    if isfield(s, group)
        t = s.(group);
        if isfield(t, name) && ~isempty(t.(name)), v = t.(name); end
    end
end

function [XTrain, YTrain, XTest, YTest] = buildFeaturesForMode(genderMap, filterGender, useMelFilter, melMode, cfg)
% Always use the original (pre-Warden) gender-split loader.
    try
        [XTrain, YTrain, XTest, YTest] = loadGenderSplitData( ...
            genderMap, filterGender, useMelFilter, melMode, cfg);
    catch
        % Back-compat: older signature without cfg
        [XTrain, YTrain, XTest, YTest] = loadGenderSplitData( ...
            genderMap, filterGender, useMelFilter, melMode);
    end
end

function maybe_plot_and_save_roc(rocInfo, tag, outDir)
% Pop up and save an ROC if rocInfo has what we need.
% Accepts either:
%   - rocInfo.fpr, rocInfo.tpr in fractions (0..1)
%   - or rocInfo.FPRpercent, rocInfo.TPR (legacy naming)
    if ~exist(outDir,'dir'), mkdir(outDir); end

    % Try to find vectors to plot
    fpr = []; tpr = [];
    if isstruct(rocInfo)
        if isfield(rocInfo,'fpr') && isfield(rocInfo,'tpr')
            fpr = rocInfo.fpr(:);  tpr = rocInfo.tpr(:);
        elseif isfield(rocInfo,'FPRpercent') && isfield(rocInfo,'TPR')
            fpr = rocInfo.FPRpercent(:)/100;  tpr = rocInfo.TPR(:);
        end
    end
    if isempty(fpr) || isempty(tpr) || numel(fpr) ~= numel(tpr) || numel(fpr) < 2
        % Nothing usable; silently return.
        return;
    end

    % Create a visible ROC figure
    fig = figure('Visible','on'); grid on; hold on;
    plot(fpr*100, tpr, 'LineWidth', 2);
    xlabel('False Positive Rate (%)'); ylabel('True Positive Rate');
    title(sprintf('ROC — %s', tag), 'Interpreter','none');
    xlim([0 5]); ylim([0 1]); % common zoom used earlier
    legend(tag, 'Interpreter','none','Location','SouthEast');

    % Save to disk as well
    outPng = fullfile(outDir, sprintf('ROC_%s.png', tag));
    outFig = fullfile(outDir, sprintf('ROC_%s.fig', tag));
    try, saveas(fig, outPng); catch, end
    try, savefig(fig, outFig); catch, end
end
% ===================================================================
