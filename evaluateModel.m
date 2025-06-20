function [accuracy, FR, FA, rocInfo] = evaluateModel(net, XTest, YTest)
    % Turn off warnings during evaluation
    warning('off', 'all');

    %% Input Validation
    if ~iscategorical(YTest)
        YTest = categorical(YTest);
    end
    YTest = YTest(:);  % Force column

    %% Get Predictions
    YPred = classify(net, XTest);
    YPred = YPred(:);

    if numel(YPred) ~= numel(YTest)
        error('Prediction/label size mismatch: %d predictions vs %d labels',...
              numel(YPred), numel(YTest));
    end

    %% Frame-Level Accuracy
    accuracy = mean(YPred == YTest) * 100;

    %% Get Posterior Scores
    scores = predict(net, XTest);  % [N x C]
    classNames = categories(YTest);

    % Assume the second class is the keyword
    if numel(classNames) < 2
        error('Expected at least 2 classes in YTest for keyword scoring.');
    end
    positiveClass = classNames{2};  % assume keyword is second class

    if ~any(strcmp(classNames, positiveClass))
        error('Positive class "%s" not found in test set.', positiveClass);
    end

    % Extract keyword scores robustly
    classIdx = find(strcmp(classNames, positiveClass));
    if isvector(scores)
        keywordScores = scores(classIdx);
    elseif size(scores, 2) == numel(classNames)
        keywordScores = scores(:, classIdx);
    else
        error('Unexpected size for score matrix: %s', mat2str(size(scores)));
    end
    keywordScores = keywordScores(:);

    if numel(keywordScores) ~= numel(YTest)
        error('Keyword score count (%d) does not match label count (%d).', ...
              numel(keywordScores), numel(YTest));
    end

    %% Posterior Smoothing
    windowSize = 15;  % 150 ms
    threshold = 0.7;
    minDuration = 5;

    smoothedPred = false(size(keywordScores));
    for i = 1:length(keywordScores) - windowSize + 1
        avg = mean(keywordScores(i:i+windowSize-1));
        if avg > threshold
            smoothedPred(i:i+windowSize-1) = true;
        end
    end

    % Suppress short activations
    state = 0;
    for i = 1:length(smoothedPred)
        if smoothedPred(i) && state == 0
            state = 1; startIdx = i;
        elseif ~smoothedPred(i) && state == 1
            if (i - startIdx) < minDuration
                smoothedPred(startIdx:i-1) = false;
            end
            state = 0;
        end
    end

    %% Keyword Metrics
    isPositive = (YTest == positiveClass);
    isNegative = ~isPositive;

    FR = sum(~smoothedPred & isPositive) / sum(isPositive) * 100;
    FA = sum(smoothedPred & isNegative) / sum(isNegative) * 100;

    %% ROC Calculation
    if any(isPositive) && any(isNegative)
        [far, frr, thresholds] = calculateROC(YTest, keywordScores, positiveClass);

        % Plot ROC
        figure;
        semilogx(far * 3600 / 0.01, frr * 100, 'LineWidth', 2);
        xlabel('False Alarms per Hour');
        ylabel('False Reject Rate (%)');
        title('ROC Curve');
        grid on;

        rocInfo.far = far;
        rocInfo.frr = frr;
        rocInfo.thresholds = thresholds;
    else
        warning('ROC not calculated - missing class examples');
        rocInfo = struct();
    end

    %% Output
    fprintf('\n=== Evaluation Results ===\n');
    fprintf('Frame Accuracy: %.2f%%\n', accuracy);
    fprintf('False Reject:   %.2f%%\n', FR);
    fprintf('False Alarm:    %.2f%%\n', FA);
    fprintf('===========================\n');

    warning('on', 'all');
end

%% Subfunction
function [far, frr, thresholds] = calculateROC(labels, scores, positiveClass)
    % Convert labels to binary: 1 for positiveClass, 0 otherwise
    yTrue = double(labels == positiveClass);
    yTrue = yTrue(:);       % Ensure column vector
    scores = scores(:);     % Ensure column vector

    % Debug shape info
    fprintf('\n==== ROC DEBUG INFO ====\n');
    fprintf('scores shape: %s\n', mat2str(size(scores)));
    fprintf('yTrue  shape: %s\n', mat2str(size(yTrue)));

    if numel(scores) ~= numel(yTrue)
        error('ROC ERROR: Score count (%d) ≠ label count (%d)', ...
            numel(scores), numel(yTrue));
    end

    % Define thresholds
    thresholds = linspace(min(scores), max(scores), 100)';
    
    % Create prediction matrix: [numThresholds × numSamples]
    pred = scores' >= thresholds;  % scores' = [1 x N], thresholds = [100 x 1]
                                  % result pred = [100 x N]

    % Broadcast yTrue: [1 x N] → [100 x N]
    yTrueMatrix = repmat(yTrue', size(pred,1), 1);  % [100 x N]

    % Debug shape confirmation
    fprintf('pred shape: %s\n', mat2str(size(pred)));
    fprintf('yTrueMatrix shape: %s\n', mat2str(size(yTrueMatrix)));

    % Confirm compatibility
    if ~isequal(size(pred), size(yTrueMatrix))
        error('Matrix shape mismatch: pred [%s] vs yTrueMatrix [%s]', ...
              mat2str(size(pred)), mat2str(size(yTrueMatrix)));
    end

    % Compute ROC
    truePos  = sum(pred & yTrueMatrix, 2);  % sum over columns
    falsePos = sum(pred & ~yTrueMatrix, 2);

    totalPos = sum(yTrue);
    totalNeg = sum(~yTrue);

    if totalPos == 0 || totalNeg == 0
        warning('ROC calculation may be invalid: positive or negative class has 0 examples');
        far = nan(size(thresholds));
        frr = nan(size(thresholds));
    else
        frr = 1 - truePos / totalPos;
        far = falsePos / totalNeg;
    end

    fprintf('==== END ROC DEBUG INFO ====\n\n');
end
