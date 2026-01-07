function [metrics, rocInfo, debug] = evaluateModel(net, XTest, YTest, makePlots, positiveLabel, fixedThreshold, cfg)

if nargin < 4 || isempty(makePlots), makePlots = false; end
if nargin < 7 || isempty(cfg), cfg = kws_config(); end

if ~iscategorical(YTest), YTest = categorical(YTest); end
YTest = YTest(:);

YPred = classify(net, XTest);
YPred = YPred(:);

% Scores aligned to net class order
S = predict(net, XTest);

netClasses = [];
try
    netClasses = net.Layers(end).Classes;
catch
    netClasses = categories(YPred);
end
netClassCell = cellstr(string(netClasses));

% Align YTest categories to network
yCats = categories(YTest);
if ~all(ismember(yCats, netClassCell))
    extras = setdiff(yCats, netClassCell);
    pref = intersect({'_unknown_','_silence_','_background_noise_'}, netClassCell, 'stable');
    if isempty(pref), fallback = netClassCell{1}; else, fallback = pref{1}; end
    YTest = mergecats(YTest, extras, fallback);
    YTest = removecats(YTest);
end
miss = setdiff(netClassCell, categories(YTest));
if ~isempty(miss), YTest = addcats(YTest, miss); end
YTest = reordercats(YTest, netClassCell);

% -------- Core metrics --------
accTop1 = mean(YPred == YTest) * 100;

% Top-k accuracy
topK = getfielddef(cfg,'eval','topK',[1 3 5]);
topkMap = struct();
for k = topK
    topkMap.(sprintf('Top%d',k)) = topKAccuracy(S, YTest, netClassCell, k) * 100;
end

% Confusion matrix
cm = [];
if isfield(cfg.eval,'confusionMatrix') && cfg.eval.confusionMatrix
    cm = confusionmat(YTest, YPred, 'Order', categorical(netClassCell));
end

% Precision/Recall/F1 (per class)
prfTbl = table();
if isfield(cfg.eval,'computePRF1') && cfg.eval.computePRF1
    prfTbl = perClassPRF1(cm, netClassCell);
end

% -------- Binary ROC for a chosen positive label --------
if nargin < 5 || isempty(positiveLabel)
    counts = arrayfun(@(c) sum(YTest==c), categorical(netClassCell));
    [~, idxDesc] = sort(counts, 'descend');
    ordered = netClassCell(idxDesc);
    maskFiller = ismember(ordered, {'_silence_','_unknown_','_background_noise_'});
    t = find(~maskFiller, 1, 'first'); if isempty(t), t = 1; end
    positiveLabel = ordered{t};
else
    if ~any(strcmp(netClassCell, positiveLabel))
        error('Requested positiveLabel "%s" is not a network class.', positiveLabel);
    end
end

classIdx = find(strcmp(netClassCell, positiveLabel), 1);
keywordScores = S(:, classIdx);

[far, frr, thresholds, AUC] = localROC(YTest, keywordScores, positiveLabel);

if nargin >= 6 && ~isempty(fixedThreshold)
    thr = fixedThreshold;
    [~, kE] = min(abs(far - frr));
else
    [~, kE] = min(abs(far - frr));
    thr = thresholds(kE);
end

posMask = (YTest == positiveLabel);
negMask = ~posMask;
posN = sum(posMask); negN = sum(negMask);

predPos = keywordScores >= thr;
FR = (sum(~predPos & posMask) / max(1,posN)) * 100;
FA = (sum( predPos & negMask) / max(1,negN)) * 100;

if makePlots && ~any(isnan(far))
    figure('Visible','on');
    plot(far * 100, frr * 100, 'LineWidth', 2); hold on;
    plot(far(kE) * 100, frr(kE) * 100, 'o','MarkerSize',6,'LineWidth',1.5);
    xlabel('False Positive Rate (%)'); ylabel('False Reject Rate (%)');
    title(sprintf('ROC (FRR vs FPR): %s (AUC=%.3f) thr=%.3f', positiveLabel, AUC, thr));
    grid on;
end

rocInfo = struct('far',far,'frr',frr,'thresholds',thresholds, ...
    'AUC',AUC,'positiveLabel',positiveLabel,'thrUsed',thr,'mode','utterance');

% -------- Multiclass OvR AUC --------
ovrTbl = table();
macroAUC = NaN;
if isfield(cfg.eval,'computeOvRAUC') && cfg.eval.computeOvRAUC
    [ovrTbl, macroAUC] = multiclassOvRAUC(YTest, S, netClassCell);
end

% Package all metrics for export
metrics = struct();
metrics.AccuracyTop1 = accTop1;
metrics.TopK = topkMap;
metrics.FR = FR;
metrics.FA = FA;
metrics.BinaryAUC = AUC;
metrics.PositiveLabel = positiveLabel;
metrics.ThresholdUsed = thr;
metrics.ConfusionMatrix = cm;
metrics.PerClassPRF1 = prfTbl;
metrics.OvRAUC = ovrTbl;
metrics.MacroOvRAUC = macroAUC;

debug = struct();
debug.netClasses = netClassCell;
debug.posCount = posN;
debug.negCount = negN;

end

% ---------- helpers ----------
function v = getfielddef(cfg, section, name, default)
v = default;
if isstruct(cfg) && isfield(cfg, section)
    S = cfg.(section);
    if isfield(S, name) && ~isempty(S.(name)), v = S.(name); end
end
end

function acc = topKAccuracy(S, Y, classCell, k)
[~, idx] = sort(S, 2, 'descend');
idx = idx(:,1:min(k,size(idx,2)));
Y = categorical(Y);
trueIdx = grp2idx(reordercats(Y, classCell));
hit = any(idx == trueIdx, 2);
acc = mean(hit);
end

function T = perClassPRF1(cm, classCell)
if isempty(cm)
    T = table(); return;
end
C = numel(classCell);
prec = zeros(C,1); rec = zeros(C,1); f1 = zeros(C,1); sup = zeros(C,1);
for i = 1:C
    tp = cm(i,i);
    fp = sum(cm(:,i)) - tp;
    fn = sum(cm(i,:)) - tp;
    sup(i) = sum(cm(i,:));
    prec(i) = tp / max(1, tp+fp);
    rec(i)  = tp / max(1, tp+fn);
    f1(i)   = 2*prec(i)*rec(i) / max(eps, (prec(i)+rec(i)));
end
T = table(classCell(:), prec, rec, f1, sup, 'VariableNames', ...
    {'Class','Precision','Recall','F1','Support'});
end

function [T, macroAUC] = multiclassOvRAUC(Y, S, classCell)
Y = reordercats(categorical(Y), classCell);
C = numel(classCell);
AUC = nan(C,1);
for i = 1:C
    lab = classCell{i};
    try
        [~,~,~,AUC(i)] = perfcurve(Y, S(:,i), lab);
    catch
        AUC(i) = NaN;
    end
end
T = table(classCell(:), AUC, 'VariableNames', {'Class','AUC_OvR'});
macroAUC = mean(AUC, 'omitnan');
end

function [far_, frr_, thr_, AUC_] = localROC(labels, scores, posLab)
labels = labels(:); scores = scores(:);
if sum(labels == posLab)==0 || sum(labels ~= posLab)==0
    far_ = nan(100,1); frr_ = nan(100,1); thr_ = nan(100,1); AUC_ = NaN; return;
end
[fpr, tpr, thr_, AUC_] = perfcurve(labels, scores, posLab);
far_ = fpr(:);
frr_ = (1 - tpr(:));
thr_ = thr_(:);
end
