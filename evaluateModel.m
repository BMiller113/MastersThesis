function varargout = evaluateModel(net, XTest, YTest, makePlots, positiveLabel, fixedThreshold)
% evaluateModel
% Supports being called with 1..5 outputs.
% Outputs (in order):
%   1) accuracy (percent)
%   2) FR (percent)
%   3) FA (percent)
%   4) rocInfo struct
%   5) debug struct

    if nargin < 4 || isempty(makePlots), makePlots = false; end
    if nargin < 5, positiveLabel = []; end
    if nargin < 6, fixedThreshold = []; end

    warning('off','all');

    if ~iscategorical(YTest), YTest = categorical(YTest); end
    YTest = YTest(:);

    % Predictions
    YPred = classify(net, XTest);
    YPred = YPred(:);
    if numel(YPred) ~= numel(YTest)
        error('Prediction/label size mismatch: %d vs %d', numel(YPred), numel(YTest));
    end

    % Get network class list/order
    try
        netClasses = net.Layers(end).Classes;   % SeriesNetwork classificationLayer
    catch
        netClasses = categories(YPred);         % fallback
    end
    netClassCell = cellstr(string(netClasses));

    % Align YTest to network classes
    yCats = categories(YTest);

    if ~all(ismember(yCats, netClassCell))
        extras = setdiff(yCats, netClassCell);
        pref = intersect({'_unknown_','_silence_','_background_noise_'}, netClassCell, 'stable');
        if isempty(pref), fallback = netClassCell{1}; else, fallback = pref{1}; end
        YTest = mergecats(YTest, extras, fallback);
        YTest = removecats(YTest);
    end

    miss = setdiff(netClassCell, categories(YTest));
    if ~isempty(miss)
        YTest = addcats(YTest, miss);
    end
    YTest = reordercats(YTest, netClassCell);

    % Accuracy
    accuracy = mean(YPred == YTest) * 100;

    % Scores
    S = predict(net, XTest);
    if size(S,2) ~= numel(netClassCell)
        error('Score width (%d) != num classes (%d).', size(S,2), numel(netClassCell));
    end

    % Choose positive label
    if isempty(positiveLabel)
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
    keywordScores = keywordScores(:);

    posMask = (YTest == positiveLabel);
    negMask = ~posMask;
    posN = sum(posMask); negN = sum(negMask);

    [far, frr, thresholds, AUC] = localROC(YTest, keywordScores, positiveLabel);

    % Threshold selection
    if ~isempty(fixedThreshold)
        thr = fixedThreshold;
        [~, k] = min(abs(far - frr));
    else
        [~, k] = min(abs(far - frr));
        thr = thresholds(k);
    end

    predPos = keywordScores >= thr;
    FR = (sum(~predPos & posMask) / max(1,posN)) * 100;
    FA = (sum( predPos & negMask) / max(1,negN)) * 100;

    if makePlots && ~any(isnan(far))
        figure('Visible','on');
        plot(far*100, frr*100, 'LineWidth', 2); hold on;
        plot(far(k)*100, frr(k)*100, 'o','MarkerSize',6,'LineWidth',1.5);
        xlabel('False Positive Rate (%)'); ylabel('False Reject Rate (%)');
        title(sprintf('ROC: %s (AUC=%.3f) | thr=%.3f', positiveLabel, AUC, thr));
        grid on;
    end

    rocInfo = struct('far',far,'frr',frr,'thresholds',thresholds, ...
        'AUC',AUC,'positiveLabel',positiveLabel,'thrUsed',thr, 'mode','utterance');

    countsVec = arrayfun(@(c) sum(YTest==c), categorical(netClassCell));
    debug = struct();
    debug.netClasses     = netClassCell;
    debug.positiveLabel  = positiveLabel;
    debug.countsTable    = table(categorical(netClassCell), countsVec(:), ...
        'VariableNames',{'Class','Count'});
    debug.posCount       = posN;
    debug.negCount       = negN;
    debug.scoreStats     = [min(keywordScores) mean(keywordScores) max(keywordScores)];

    warning('on','all');

    % varargout (supports 1..5 outputs)
    outs = {accuracy, FR, FA, rocInfo, debug};
    nreq = max(1, nargout);
    nreq = min(nreq, numel(outs));
    varargout = outs(1:nreq);

end

% =================== local helper ===================
function [far_, frr_, thr_, AUC_] = localROC(labels, scores, posLab)
    labels = labels(:); scores = scores(:);
    if sum(labels == posLab)==0 || sum(labels ~= posLab)==0
        warning('ROC ill-defined: missing positives or negatives.');
        far_ = nan(100,1); frr_ = nan(100,1); thr_ = nan(100,1); AUC_ = NaN; return;
    end
    try
        [fpr, tpr, thr_, AUC_] = perfcurve(labels, scores, posLab);
        far_ = fpr(:); frr_ = (1 - tpr(:)); thr_ = thr_(:);
    catch
        thr_ = linspace(min(scores), max(scores), 200)';  % manual sweep
        pred = scores' >= thr_;
        y = double(labels==posLab)';  % 1xN
        tp = sum(pred & (y==1), 2); fp = sum(pred & (y==0), 2);
        P = sum(y==1); N = sum(y==0);
        tpr = tp / max(1,P); fpr = fp / max(1,N);
        far_ = fpr(:); frr_ = (1 - tpr(:));
        AUC_ = trapz(fpr, tpr);
    end
end
