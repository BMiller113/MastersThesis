% ============================== main.m ==============================
% Keyword Spotting main runner
%
% Purpose
%   This script is the primary experiment runner for the KWS project.
%   It trains and evaluates CNN keyword-spotting models on Google Speech
%   Commands (V1 or V2).
%
% What it does
%   For each requested experimental condition (mode × melMode):
%     1) Loads train/test file lists from the active dataset (V1 or V2)
%     2) Builds log-mel features into a fixed tensor [B x W x 1 x N]
%     3) Trains the configured CNN architecture
%     4) Evaluates performance (accuracy + ROC-derived metrics)
%     5) Exports models/metrics/ROC plots in a consistent legacy schema
%
% Outputs (written under cfg.paths.outputDir)
%   models/
%     model_<tag>.mat
%
%   metrics/
%     metrics_<tag>.mat
%     metrics_<tag>.csv
%     auc_over_<tag>.csv
%     prf1_<tag>.csv
%
%   roc/
%     <tag>.fig / <tag>.png   (per-run ROC)
%     overlay/
%       <overlayName>.fig / <overlayName>.png  (overlay ROC)
%
% Legacy compatibility
%   The metrics CSV schema below matches the older spreadsheet-friendly
%   format so results from different project phases remain comparable.
%
%   metrics_<tag>.csv columns:
%     RunKey, Dataset, Arch, Tag, Top1Acc, Top3Acc, Top5Acc, FR, FA,
%     BinaryAUC, MacroOvRAUC, Params, MACs
%
% IMPORTANT NOTES
%   - This runner disables figure popups to avoid interrupting while on 
%     long runs. ROC plots are still generated and saved to disk.
%   - "MACs" here are CNN forward-pass MACs only (no audio preprocessing).
%   - The run naming uses "intended" geometry from cfg (B_name/W_name) to
%     preserve historical naming convention, even if some mel modes produce
%     a different actual bin count (e.g. prop8k producing 48 bins).
%
% ====================================================================

clear; clc;

cfg = kws_config();

% UI control: disable popups

set(0,'DefaultFigureVisible','off');

% Output folders
outRoot    = cfg.paths.outputDir;
modelsDir  = fullfile(outRoot,'models');
metricsDir = fullfile(outRoot,'metrics');
rocDir     = fullfile(outRoot,'roc');              
overlayDir = fullfile(rocDir,'overlay');

ensureDir(outRoot);
ensureDir(modelsDir);
ensureDir(metricsDir);
ensureDir(rocDir);
ensureDir(overlayDir);

% Dataset version (v1/v2) 
% The dataset selection is controlled by cfg.dataset.version.
% Expected values: "v1" or "v2"
if ~isfield(cfg,'dataset') || ~isfield(cfg.dataset,'version') || isempty(cfg.dataset.version)
    cfg.dataset.version = "v2";
end
dsTag = lower(string(cfg.dataset.version));   % "v1" or "v2"

% Architecture
% This is the model family being tested (one per run of main.m).
% Examples: 'trad-fpool3', 'tpool2', 'one-fstride4'
ARCH    = string(cfg.model.arch);
archTok = sanitizeArchTok(ARCH);              % stored token (hyphens -> underscores)

% Intended geometry for naming
% This is the "canonical" frequency/time shape used in filenames.
% The tensor shape actually produced by extractFeatures may differ in some
% mel modes (notably proportional-band modes like prop8k). Naming
% based on cfg for consistency with older results.
B_name = getf(cfg,'features','baseBands',40);
W_name = getf(cfg,'features','targetFrames',32);

% Overlay label: one overlay per dataset + architecture + geometry.
overlayNameForThisBatch = sprintf('%s__%s__%dx%d', dsTag, archTok, B_name, W_name);

% The "genderModes" list historically includes several variants. In current
% version, only these are supported:
%   - "none"     : default mel features
%   - "mel-only" : iterate mel variants (default/narrow/wide/prop7k/prop8k/etc.)
genderModes = cfg.experiments.includeModes;
melModesAll = cfg.experiments.melModes;

% Store ROC curves from successful conditions and then generate a
% final overlay plot at the end.
overlayCurves = {};
overlayLabels = {};

for g = 1:numel(genderModes)
    runMode = lower(string(genderModes{g}));

    % Map high-level modes into flags used by this runner.
    % modeTok is used in file naming to match prior conventions.
    switch runMode
        case "none"
            useMelFilter  = false;
            localMelModes = {'default'};
            modeTok = "none";
        case "mel-only"
            useMelFilter  = true;
            localMelModes = melModesAll;
            modeTok = "mel_only";
        otherwise
            warning('Skipping unknown run mode: %s', runMode);
            continue;
    end

    for m = 1:numel(localMelModes)
        melMode = lower(string(localMelModes{m}));

        % When useMelFilter is false, only run default front end.
        if ~useMelFilter && melMode ~= "default"
            continue;
        end

        fprintf('\n=== %s | MEL=%s | GROUP=ALL | DS=%s ===\n', ...
            upper(runMode), upper(melMode), upper(dsTag));

        % Load dataset
        % loadAudioData(cfg) should:
        %   - pick dataset root based on cfg.dataset.version
        %   - return train/test file lists using testing_list.txt and validation_list.txt
        try
            [trainFiles, trainLabels, testFiles, testLabels] = loadAudioData(cfg);
        catch ME
            warning('Data load failed for %s/%s/all/%s: %s', runMode, melMode, dsTag, ME.message);
            continue;
        end
        fprintf('[%s] Train=%d Test=%d\n', upper(dsTag), numel(trainFiles), numel(testFiles));

        % Feature extraction
        % Features are 4-D: [freqBins x timeFrames x 1 x N]
        % vTr/vTe indicate files that were successfully processed.
        tFeat = tic;
        if useMelFilter
            [XTrain, vTr] = extractFeatures(trainFiles,'all',char(melMode),cfg);
            [XTest,  vTe] = extractFeatures(testFiles, 'all',char(melMode),cfg);
        else
            [XTrain, vTr] = extractFeatures(trainFiles,'all','default',cfg);
            [XTest,  vTe] = extractFeatures(testFiles, 'all','default',cfg);
        end
        YTrain = trainLabels(vTr);
        YTest  = testLabels(vTe);

        fprintf('Features built in %.1f s\n', toc(tFeat));

        % After filtering invalid files, ensure there are still have multiple classes.
        if numel(categories(YTrain)) < 2 || numel(categories(YTest)) < 2
            warning('Not enough classes after filtering; skipping this condition.');
            continue;
        end

        % True tensor geometry (for network input & MACs)
        % This is the real input size the network will see, which may differ
        % from the "intended" cfg size in certain mel modes.
        B_actual = size(XTrain,1);
        W_actual = size(XTrain,2);

        frameMs = getf(cfg,'features','frameMs',25);
        hopMs   = getf(cfg,'features','hopMs',10);
        spanMs  = frameMs + (W_actual-1)*hopMs;
        fprintf('CNN input (actual tensor): %d×%d×1 (~%.0f ms)\n', B_actual, W_actual, spanMs);

        % Train
        fprintf('Training %s...\n', char(ARCH));

        % defineCNNArchitecture is written to accept variable input sizes.
        layers = defineCNNArchitecture(numel(categories(YTrain)), char(ARCH), B_actual, W_actual);

        % trainCNN handles GPU, holdout, and core training options.
        net = trainCNN(XTrain, YTrain, layers, cfg);
        fprintf('Training complete.\n');

        % Evaluate (no plotting)
        fprintf('Evaluating...\n');
        makePlots = false;
        [Top1Acc, FR, FA, rocInfo] = callEvaluateModelRobust(net, XTest, YTest, makePlots, cfg);

        % Scores for TopK + MacroAUC 
        % Scores come in network class order. Build a class list and align
        % YTest to ensure consistent indexing.
        [~, scores] = classify(net, XTest);              % scores: N x C
        classList = netClassList(net, YTrain, scores);   % cellstr class names (network order)
        YTestAligned = alignCats(YTest, classList);      % enforce same order

        % Top-K accuracy
        [Top3Acc, Top5Acc] = computeTopKAcc(scores, YTestAligned, 3, 5);

        % Binary AUC
        % Binary AUC is one-vs-rest AUC for a chosen positive keyword.
        % If cfg.experiments.forcePosLabel is set, use it; otherwise select
        % the most common non-filler label.
        posLab = pickPositiveLabel(cfg, YTestAligned, classList);
        BinaryAUC = computeBinaryAUC(scores, YTestAligned, classList, posLab);

        % Macro OvR AUC
        % MacroOvRAUC = average AUC over all classes (one-vs-rest), ignoring
        % classes that are absent in the test set.
        MacroOvRAUC = computeMacroOvRAUC(scores, YTestAligned, classList);

        % Params + MACs
        % Params = number of learnable weights (including BN scale/offset).
        % MACs   = CNN forward-pass MACs (conv + FC), excluding preprocessing (!!!).
        Params = countLearnableParams(net);  
        MACs   = NaN;
        try
            MACs = double(estimateMACs(net.Layers, [B_actual W_actual 1]));
        catch
            MACs = NaN;
        end

        % Per-class PRF1
        % Exports a per-class precision/recall/F1 table for diagnostics.
        YPred = classify(net, XTest);
        prfTbl = perClassPRF1(YTestAligned, YPred);

        % Tag naming
        % Keep the legacy tag schema to preserve compatibility with older
        % results folders and spreadsheets.
        %
        % Example:
        %   v2__one_fstride4__mel_only_default_all__40x98
        condTok = sprintf('%s_%s_all', modeTok, melMode);  % mel_only_default_all
        tag = sprintf('%s__%s__%s__%dx%d', dsTag, archTok, condTok, B_name, W_name);

        % Save model + metrics MAT
        modelPath   = fullfile(modelsDir,  ['model_'   tag '.mat']);
        metricsPath = fullfile(metricsDir, ['metrics_' tag '.mat']);

        metrics.RunKey      = tag;
        metrics.Dataset     = char(upper(dsTag));
        metrics.Arch        = char(ARCH);
        metrics.Tag         = tag;

        metrics.Top1Acc     = double(Top1Acc);
        metrics.Top3Acc     = double(Top3Acc);
        metrics.Top5Acc     = double(Top5Acc);

        metrics.FR          = double(FR);
        metrics.FA          = double(FA);

        metrics.BinaryAUC   = double(BinaryAUC);
        metrics.MacroOvRAUC = double(MacroOvRAUC);

        metrics.Params      = double(Params);
        metrics.MACs        = double(MACs);

        save(modelPath,   'net','cfg','-v7.3');
        save(metricsPath, 'metrics','rocInfo','cfg','-v7.3');

        fprintf('Top1=%.2f%% | Top3=%.2f%% | Top5=%.2f%% | FR=%.2f%% | FA=%.2f%% | BinAUC=%.4f | MacroAUC=%.4f\n', ...
            Top1Acc, Top3Acc, Top5Acc, FR, FA, BinaryAUC, MacroOvRAUC);
        fprintf('Params=%.0f\n', Params);
        if ~isnan(MACs), fprintf('MACs=%.3e\n', MACs); end

        % Save per-run ROC files into Results/roc
        % saveRoc_TPR_NoPopup converts rocInfo's FAR/FRR into a TPR-vs-FPR ROC plot
        % and writes it as .fig and .png.
        savedRoc = saveRoc_TPR_NoPopup(rocInfo, tag, rocDir, cfg);

        % If curve saved successfully, include it in the overlay pool.
        if savedRoc
            overlayCurves{end+1} = rocInfo;
            overlayLabels{end+1} = sprintf('%s - %s - %s', upper(modeTok), upper(melMode), upper(dsTag)); %#ok<SAGROW>
        else
            warning('ROC not saved for %s (rocInfo missing far/frr).', tag);
        end

        % Export CSV artifacts
        %   metrics_<tag>.csv
        %   auc_over_<tag>.csv
        %   prf1_<tag>.csv
        try
            exportMetricsCSV_Legacy(metricsDir, tag, metrics);
            exportAUCOverCSV(metricsDir, tag, rocInfo, metrics);
            exportPRF1CSV(metricsDir, tag, prfTbl);
        catch ME
            warning('CSV export failed for %s: %s', tag, ME.message);
        end

    end
end

% Overlay ROC 
% After the loop, generate a single overlay plot comparing the ROC curves
% from all successfully-evaluated conditions in this batch.
if ~isempty(overlayCurves)
    saveOverlayRoc_TPR_NoPopup(overlayCurves, overlayLabels, overlayNameForThisBatch, overlayDir, cfg);
else
    warning('No curves collected for overlay.');
end

fprintf('\nDone.\n');

% ============================ Helpers ============================

function archTok = sanitizeArchTok(archStr)
    % Converts an architecture string into a token suitable for filenames.
    % Example: "one-fstride4" -> "one_fstride4"
    archTok = lower(string(archStr));
    archTok = strrep(archTok, '-', '_');
    archTok = char(archTok);
end

function [acc,FR,FA,rocInfo] = callEvaluateModelRobust(net, XTest, YTest, makePlots, cfg)
    % evaluateModel has had multiple formats over the life of the project.
    % This wrapper detects how many outputs evaluateModel provides and returns
    % a consistent subset.
    %
    % Expected (modern) signature:
    %   [acc, FR, FA, rocInfo] = evaluateModel(net, XTest, YTest, makePlots, positiveLabel, fixedThreshold)

    posLab = [];
    thr = [];
    if isfield(cfg,'experiments')
        if isfield(cfg.experiments,'forcePosLabel'), posLab = cfg.experiments.forcePosLabel; end
        if isfield(cfg.experiments,'fixedThreshold'), thr = cfg.experiments.fixedThreshold; end
    end

    nout = nargout('evaluateModel');
    switch nout
        case 1
            acc = evaluateModel(net, XTest, YTest, makePlots, posLab, thr);
            FR = NaN; FA = NaN; rocInfo = struct();
        case 2
            [acc,FR] = evaluateModel(net, XTest, YTest, makePlots, posLab, thr);
            FA = NaN; rocInfo = struct();
        case 3
            [acc,FR,FA] = evaluateModel(net, XTest, YTest, makePlots, posLab, thr);
            rocInfo = struct();
        otherwise
            [acc,FR,FA,rocInfo] = evaluateModel(net, XTest, YTest, makePlots, posLab, thr);
    end
end

function cls = netClassList(net, YTrain, scores)
    % Build a class label list in the same order as the score columns.
    % The ideal source is net.Layers(end).Classes, but fallbacks are included.
    try
        cls = cellstr(string(net.Layers(end).Classes));
    catch
        cls = cellstr(string(categories(YTrain)));
    end

    % If the list length doesn't match the score matrix, fall back again.
    if numel(cls) ~= size(scores,2)
        cls = cellstr(string(categories(YTrain)));
        if numel(cls) ~= size(scores,2)
            cls = arrayfun(@(i) sprintf('c%d', i), 1:size(scores,2), 'UniformOutput', false);
        end
    end
end

function Y = alignCats(Y, classList)
    % Ensure Y is categorical with categories ordered to match classList.
    if ~iscategorical(Y), Y = categorical(Y); end
    Y = Y(:);
    miss = setdiff(classList, categories(Y));
    if ~isempty(miss), Y = addcats(Y, miss); end
    Y = reordercats(Y, classList);
end

function [topK3, topK5] = computeTopKAcc(scores, YTrue, k3, k5)
    % Compute Top-K accuracy using score ranking.
    % k3 and k5.
    [~, idx] = sort(scores, 2, 'descend');
    trueIdx = grp2idx(YTrue);

    top3 = idx(:,1:min(k3,size(idx,2)));
    top5 = idx(:,1:min(k5,size(idx,2)));

    topK3 = 100 * mean(any(top3 == trueIdx, 2));
    topK5 = 100 * mean(any(top5 == trueIdx, 2));
end

function posLab = pickPositiveLabel(cfg, YTest, classList)
    % Choose a "positive label" for binary ROC/AUC.
    % If cfg.experiments.forcePosLabel is set and valid, use it.
    % Otherwise choose the most frequent non-filler label in YTest.
    posLab = [];
    if isfield(cfg,'experiments') && isfield(cfg.experiments,'forcePosLabel') && ~isempty(cfg.experiments.forcePosLabel)
        posLab = char(cfg.experiments.forcePosLabel);
        if isstring(posLab), posLab = char(posLab); end
        if ~any(strcmp(classList, posLab))
            posLab = [];
        end
    end
    if isempty(posLab)
        counts = zeros(numel(classList),1);
        for i = 1:numel(classList)
            counts(i) = sum(YTest == classList{i});
        end
        [~, ord] = sort(counts,'descend');
        filler = {'_silence_','_unknown_','_background_noise_'};
        pick = ord(find(~ismember(classList(ord), filler), 1, 'first'));
        if isempty(pick), pick = ord(1); end
        posLab = classList{pick};
    end
end

function auc = computeBinaryAUC(scores, YTest, classList, posLab)
    % One-vs-rest AUC for a selected positive label.
    auc = NaN;
    j = find(strcmp(classList, posLab), 1);
    if isempty(j), return; end
    s = scores(:,j);

    try
        [~,~,~,auc] = perfcurve(YTest, s, posLab);
    catch
        try
            [fpr,tpr] = localROC(YTest, s, posLab);
            auc = trapz(fpr, tpr);
        catch
            auc = NaN;
        end
    end
    auc = double(auc);
end

function macro = computeMacroOvRAUC(scores, YTest, classList)
    % Macro-average one-vs-rest AUC across classes.
    aucs = nan(numel(classList),1);
    for i = 1:numel(classList)
        lab = classList{i};
        s = scores(:,i);
        if sum(YTest==lab)==0 || sum(YTest~=lab)==0
            continue;
        end
        try
            [~,~,~,aucs(i)] = perfcurve(YTest, s, lab);
        catch
            try
                [fpr,tpr] = localROC(YTest, s, lab);
                aucs(i) = trapz(fpr, tpr);
            catch
                aucs(i) = NaN;
            end
        end
    end
    macro = double(mean(aucs,'omitnan'));
end

function [fpr, tpr] = localROC(labels, scores, posLab)
    % Manual ROC computation (fallback if perfcurve is unavailable).
    labels = labels(:); scores = scores(:);
    thr = unique(scores);
    thr = sort(thr,'descend');
    P = sum(labels==posLab);
    N = sum(labels~=posLab);
    tpr = zeros(numel(thr),1);
    fpr = zeros(numel(thr),1);
    for k = 1:numel(thr)
        predPos = scores >= thr(k);
        tp = sum(predPos & (labels==posLab));
        fp = sum(predPos & (labels~=posLab));
        tpr(k) = tp / max(1,P);
        fpr(k) = fp / max(1,N);
    end
    [fpr, ord] = sort(fpr);
    tpr = tpr(ord);
end

function p = countLearnableParams(net)
    % Robust parameter counter that works across MATLAB versions and
    % different network container types.
    p = 0;

    try
        layers = net.Layers;
    catch
        p = NaN;
        return;
    end

    for i = 1:numel(layers)
        L = layers(i);

        % Common learnables
        if isprop(L,'Weights') && ~isempty(L.Weights), p = p + numel(L.Weights); end
        if isprop(L,'Bias')    && ~isempty(L.Bias),    p = p + numel(L.Bias);    end

        % BatchNorm learnables
        if isprop(L,'Offset')  && ~isempty(L.Offset),  p = p + numel(L.Offset);  end
        if isprop(L,'Scale')   && ~isempty(L.Scale),   p = p + numel(L.Scale);   end
    end

    p = double(p);
end

function ok = saveRoc_TPR_NoPopup(rocInfo, baseName, outDir, cfg)
    % Save ROC plot (TPR vs FPR) without opening a visible window.
    % rocInfo is expected to contain:
    %   rocInfo.far (FPR)
    %   rocInfo.frr (FRR)
    ok = false;
    if ~isstruct(rocInfo) || ~isfield(rocInfo,'far') || ~isfield(rocInfo,'frr'), return; end

    FPR = rocInfo.far(:);
    FRR = rocInfo.frr(:);
    if isempty(FPR) || isempty(FRR) || numel(FPR) ~= numel(FRR), return; end
    if any(~isfinite(FPR)) || any(~isfinite(FRR)), return; end

    TPR = 1 - FRR;

    fig = figure('Visible','off');
    plot(FPR*100, TPR*100, 'LineWidth', 2);
    grid on;
    xlabel('False Positive Rate (%)');
    ylabel('True Positive Rate (%)');
    title(baseName, 'Interpreter','none');

    xlim(getf_roc(cfg,'xlim',[0 5]));
    ylim([0 100]);

    saveas(fig, fullfile(outDir, [baseName '.png']));
    savefig(fig, fullfile(outDir, [baseName '.fig']));
    close(fig);

    ok = true;
end

function saveOverlayRoc_TPR_NoPopup(curves, labels, baseName, outDir, cfg)
    % Save overlay ROC plot comparing multiple conditions.
    fig = figure('Visible','off'); hold on; grid on;

    for i = 1:numel(curves)
        rc = curves{i};
        if ~isfield(rc,'far') || ~isfield(rc,'frr'), continue; end
        FPR = rc.far(:);
        TPR = 1 - rc.frr(:);
        if isempty(FPR) || isempty(TPR) || numel(FPR) ~= numel(TPR), continue; end
        plot(FPR*100, TPR*100, 'LineWidth', 2);
    end

    xlabel('False Positive Rate (%)');
    ylabel('True Positive Rate (%)');
    title(sprintf('ROC Overlay -- %s', baseName), 'Interpreter','none');

    if ~isempty(labels)
        legend(labels, 'Interpreter','none', 'Location','best');
    end

    xlim(getf_roc(cfg,'xlim',[0 5]));
    ylim([0 100]);

    saveas(fig, fullfile(outDir, [baseName '.png']));
    savefig(fig, fullfile(outDir, [baseName '.fig']));
    close(fig);
end

function exportMetricsCSV_Legacy(metricsDir, tag, metrics)
    % Export a one-row CSV
    out = fullfile(metricsDir, ['metrics_' tag '.csv']);

    T = table( ...
        string(metrics.RunKey), ...
        string(metrics.Dataset), ...
        string(metrics.Arch), ...
        string(metrics.Tag), ...
        metrics.Top1Acc, metrics.Top3Acc, metrics.Top5Acc, ...
        metrics.FR, metrics.FA, ...
        metrics.BinaryAUC, metrics.MacroOvRAUC, ...
        metrics.Params, metrics.MACs, ...
        'VariableNames', {'RunKey','Dataset','Arch','Tag','Top1Acc','Top3Acc','Top5Acc','FR','FA','BinaryAUC','MacroOvRAUC','Params','MACs'});

    writetable(T, out);
end

function exportAUCOverCSV(metricsDir, tag, rocInfo, metrics)
    % Export the ROC curve points (FPR/TPR) and threshold (if available).
    out = fullfile(metricsDir, ['auc_over_' tag '.csv']);

    if ~isstruct(rocInfo) || ~isfield(rocInfo,'far') || ~isfield(rocInfo,'frr')
        T = table([],[],[],[], 'VariableNames', {'FPR','TPR','Threshold','AUC'});
        writetable(T, out);
        return;
    end

    fpr = rocInfo.far(:);
    frr = rocInfo.frr(:);
    tpr = 1 - frr;

    thr = NaN(size(fpr));
    if isfield(rocInfo,'thresholds') && ~isempty(rocInfo.thresholds)
        thr0 = rocInfo.thresholds(:);
        if numel(thr0) == numel(fpr)
            thr = thr0;
        end
    end

    auc = metrics.BinaryAUC;
    if isnan(auc) && isfield(rocInfo,'AUC') && ~isempty(rocInfo.AUC)
        auc = double(rocInfo.AUC);
    end

    AUCcol = repmat(auc, size(fpr));
    T = table(fpr, tpr, thr, AUCcol, 'VariableNames', {'FPR','TPR','Threshold','AUC'});
    writetable(T, out);
end

function exportPRF1CSV(metricsDir, tag, prfTbl)
    % Export per-class precision/recall/F1 in a consistent CSV.
    out = fullfile(metricsDir, ['prf1_' tag '.csv']);
    writetable(prfTbl, out);
end

function T = perClassPRF1(YTrue, YPred)
    % Compute per-class precision/recall/F1 and support counts.
    if ~iscategorical(YTrue), YTrue = categorical(YTrue); end
    if ~iscategorical(YPred), YPred = categorical(YPred); end
    YTrue = YTrue(:); YPred = YPred(:);

    cats = union(categories(YTrue), categories(YPred), 'stable');
    YTrue = categorical(YTrue, cats);
    YPred = categorical(YPred, cats);

    C = confusionmat(YTrue, YPred, 'Order', categorical(cats));
    support = sum(C,2);

    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2)  - TP;

    prec = TP ./ max(1, TP + FP);
    rec  = TP ./ max(1, TP + FN);
    f1   = 2 .* (prec .* rec) ./ max(eps, (prec + rec));

    T = table(categorical(cats), prec, rec, f1, support, ...
        'VariableNames', {'Class','Precision','Recall','F1','Support'});
end

function ensureDir(p)
    % Create output directories if they do not exist.
    if ~exist(p,'dir'), mkdir(p); end
end

function v = getf(cfg, section, name, defaultV)
    % Safe config getter for nested structs.
    v = defaultV;
    if isfield(cfg, section)
        S = cfg.(section);
        if isfield(S, name) && ~isempty(S.(name)), v = S.(name); end
    end
end

function v = getf_roc(cfg, field, defaultV)
    % Read ROC plotting preferences from cfg.plots.roc.* if present.
    v = defaultV;
    if isfield(cfg,'plots') && isfield(cfg.plots,'roc')
        R = cfg.plots.roc;
        if isfield(R, field) && ~isempty(R.(field)), v = R.(field); end
    end
end

function macs = estimateMACs(layers, inputSizeHW1)
    % Estimate CNN forward-pass MACs (conv + FC). Pooling affects output
    % geometry but does not contribute MACs here.
    H = inputSizeHW1(1);
    W = inputSizeHW1(2);
    C = inputSizeHW1(3);

    macs = 0;

    for i = 1:numel(layers)
        L = layers(i);

        if isa(L,'nnet.cnn.layer.ImageInputLayer')
            % No compute contribution.
        elseif isa(L,'nnet.cnn.layer.Convolution2DLayer')
            k = L.FilterSize;
            Cout = L.NumFilters;
            stride = L.Stride;
            pad = L.PaddingSize;

            [Hout, Wout] = outSize2D(H, W, k(1), k(2), stride(1), stride(2), pad, L.PaddingMode);
            macs = macs + double(Hout) * double(Wout) * double(Cout) * double(k(1)*k(2)*C);

            % Update current tensor shape for subsequent layers.
            H = Hout; W = Wout; C = Cout;

        elseif isa(L,'nnet.cnn.layer.MaxPooling2DLayer') || isa(L,'nnet.cnn.layer.AveragePooling2DLayer')
            % Pooling changes geometry but is treated as negligible compute here.
            p = L.PoolSize; s = L.Stride; pad = L.PaddingSize;
            [Hout, Wout] = outSize2D(H, W, p(1), p(2), s(1), s(2), pad, L.PaddingMode);
            H = Hout; W = Wout;

        elseif isa(L,'nnet.cnn.layer.GlobalAveragePooling2DLayer')
            H = 1; W = 1;

        elseif isa(L,'nnet.cnn.layer.FullyConnectedLayer')
            out = L.OutputSize;
            in  = H * W * C;
            macs = macs + double(in) * double(out);
            H = 1; W = 1; C = out;
        end
    end
end

function [Hout, Wout] = outSize2D(H, W, kH, kW, sH, sW, pad, padMode)
    % Compute output height/width for conv/pooling.
    % For "same" padding, MATLAB uses ceil(H/stride).
    if isempty(pad), pad = 0; end
    if isscalar(pad)
        pt = pad; pb = pad; pl = pad; pr = pad;
    else
        pt = pad(1); pb = pad(2); pl = pad(3); pr = pad(4);
    end

    switch lower(char(padMode))
        case 'same'
            Hout = ceil(H / sH);
            Wout = ceil(W / sW);
        otherwise
            Hout = floor((H + pt + pb - kH) / sH) + 1;
            Wout = floor((W + pl + pr - kW) / sW) + 1;
    end

    Hout = max(1, Hout);
    Wout = max(1, Wout);
end
