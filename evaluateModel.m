function [accuracy, FR, FA, rocInfo, debug] = evaluateModel(net, XTest, YTest, makePlots, positiveLabel, fixedThreshold)

    if nargin < 4 || isempty(makePlots), makePlots = false; end
    warning('off','all');

    % Ensure categorical column
    if ~iscategorical(YTest), YTest = categorical(YTest); end
    YTest = YTest(:);

    % Predictions (already in the network's class space)
    YPred = classify(net, XTest);
    YPred = YPred(:);
    if numel(YPred) ~= numel(YTest)
        error('Prediction/label size mismatch: %d vs %d', numel(YPred), numel(YTest));
    end

    % --- Get network class list/order ---
    netClasses = [];
    try
        netClasses = net.Layers(end).Classes;   % SeriesNetwork classificationLayer
    catch
        netClasses = categories(YPred);         % fallback
    end
    netClassCell = cellstr(string(netClasses)); % normalize to cellstr

    % Align YTest to network classes
    yCats = categories(YTest);

    % 1) If YTest has classes the net doesn't know, merge them to a fallback the net does know
    if ~all(ismember(yCats, netClassCell))
        extras = setdiff(yCats, netClassCell);    % present in YTest, absent in net
        % choose fallback: prefer
        % '_unknown_'/'_silence_'/'_background_noise_' if available  (!!!)
        pref = intersect({'_unknown_','_silence_','_background_noise_'}, netClassCell, 'stable');
        if isempty(pref), fallback = netClassCell(1); else, fallback = pref{1}; end
        YTest = mergecats(YTest, extras, fallback);  % merge all 'extras' into fallback
        YTest = removecats(YTest);
    end

    % 2) Add missing categories (present in net, absent in YTest) so reordercats can work
    miss = setdiff(netClassCell, categories(YTest));
    if ~isempty(miss)
        YTest = addcats(YTest, miss);
    end

    % 3) Reorder to match network order
    YTest = reordercats(YTest, netClassCell);

    % Accuracy (after alignment)
    accuracy = mean(YPred == YTest) * 100;

    % Scores in the network's class order
    S = predict(net, XTest);  % [N x C]
    if size(S,2) ~= numel(netClassCell)
        error('Prediction score matrix width (%d) != network classes (%d).', size(S,2), numel(netClassCell));
    end

    % Choose positive label
    if nargin < 5 || isempty(positiveLabel)
        % pick the most frequent non-filler label in aligned YTest
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

    % Binary scores for chosen keyword
    classIdx = find(strcmp(netClassCell, positiveLabel), 1);
    keywordScores = S(:, classIdx);
    keywordScores = keywordScores(:);

    posMask = (YTest == positiveLabel);
    negMask = ~posMask;
    posN = sum(posMask); negN = sum(negMask);

    % ROC + threshold
    [far, frr, thresholds, AUC] = localROC(YTest, keywordScores, positiveLabel);

    if nargin >= 6 && ~isempty(fixedThreshold)
        thr = fixedThreshold;
    else
        [~, k] = min(abs(far - frr));   % EER threshold
        thr = thresholds(k);
    end

    % FR / FA (%)
    predPos = keywordScores >= thr;
    FR = (sum(~predPos & posMask) / max(1,posN)) * 100;
    FA = (sum( predPos & negMask) / max(1,negN)) * 100;

    if makePlots && ~any(isnan(far))
        figure('Visible','on');
        semilogx(far * 3600 / 0.01, frr * 100, 'LineWidth', 2); hold on;
        [~, k] = min(abs(far - frr));
        plot(far(k) * 3600 / 0.01, frr(k) * 100, 'o','MarkerSize',6,'LineWidth',1.5);
        xlabel('False Alarms per Hour'); ylabel('False Reject Rate (%)');
        title(sprintf('ROC: %s (AUC=%.3f) | thr=%.3f', positiveLabel, AUC, thr));
        grid on;
    end

    rocInfo = struct('far',far,'frr',frr,'thresholds',thresholds, ...
                     'AUC',AUC,'positiveLabel',positiveLabel,'thrUsed',thr);

    % Debug
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

    % Helpers
    function [far_, frr_, thr_, AUC_] = localROC(labels, scores, posLab)
        labels = labels(:); scores = scores(:);
        if sum(labels == posLab)==0 || sum(labels ~= posLab)==0
            warning('ROC ill-defined: missing positives or negatives.');
            far_ = nan(100,1); frr_ = nan(100,1); thr_ = nan(100,1); AUC_ = NaN; return;
        end
        try
            [fpr, tpr, thr_, AUC_] = perfcurve(labels, scores, posLab);
            far_ = fpr; frr_ = 1 - tpr;
            far_ = far_(:); frr_ = frr_(:); thr_ = thr_(:);
        catch
            thr_ = linspace(min(scores), max(scores), 200)';  % manual sweep
            pred = scores' >= thr_;
            y = double(labels==posLab)';  % 1xN
            tp = sum(pred & (y==1), 2); fp = sum(pred & (y==0), 2);
            P = sum(y==1); N = sum(y==0);
            tpr = tp / max(1,P); fpr = fp / max(1,N);
            far_ = fpr; frr_ = 1 - tpr;
            AUC_ = trapz(fpr, tpr);
        end
    end
end
