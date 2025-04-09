function [accuracy, FR, FA] = evaluateModel(net, XTest, YTest)
    % Turn off warnings during evaluation
    warning('off', 'all');
    
    % Get predictions
    YPred = classify(net, XTest);
    scores = predict(net, XTest);
    classNames = categories(YTest);
    positiveClass = classNames{2}; % Assuming first class is keyword
    
    %% Posterior Handling (Sliding Window Smoothing)
    % Parameters from paper
    windowSize = 15; % Frames (~150ms for 10ms frame shift)
    threshold = 0.7; % Confidence threshold
    minDuration = 5; % Minimum consecutive detections
    
    % Initialize smoothed predictions
    smoothedPred = false(size(scores,1),1);
    keywordScores = scores(:,strcmp(classNames, positiveClass));
    
    % Sliding window average
    for i = 1:length(keywordScores)-windowSize+1
        windowAvg = mean(keywordScores(i:i+windowSize-1));
        if windowAvg > threshold
            smoothedPred(i:i+windowSize-1) = true;
        end
    end
    
    % Remove brief false activations (HMM-like smoothing)
    state = 0;
    for i = 1:length(smoothedPred)
        if smoothedPred(i) && state == 0
            state = 1;
            startIdx = i;
        elseif ~smoothedPred(i) && state == 1
            if (i - startIdx) < minDuration
                smoothedPred(startIdx:i-1) = false; % Remove short detections
            end
            state = 0;
        end
    end
    
    %% Calculate Metrics
    % Frame-level accuracy (original)
    accuracy = mean(YPred == YTest) * 100;
    
    % Keyword-level metrics (smoothed)
    isPositive = (YTest == positiveClass);
    isNegative = ~isPositive;
    
    FR = sum(~smoothedPred & isPositive) / sum(isPositive) * 100;
    FA = sum(smoothedPred & isNegative) / sum(isNegative) * 100;
    
    %% Output
    fprintf('\n=== Final Evaluation ===\n');
    fprintf('Frame Accuracy:  %.2f%%\n', accuracy);
    fprintf('Keyword Detection:\n');
    fprintf('False Reject:    %.2f%%\n', FR);
    fprintf('False Alarm:     %.2f%%\n\n', FA);
    
    % Optional: Plot confidence over time for first test sample
    if ~isempty(XTest)
        figure;
        plot(keywordScores(1:min(300,end)), 'b-', 'LineWidth', 1.5); % First 3 sec
        hold on;
        plot([1 length(keywordScores)], [threshold threshold], 'r--');
        ylim([0 1]);
        title('Keyword Confidence Scores');
        xlabel('Frame (10ms)');
        ylabel('Confidence');
        legend('Score', 'Threshold', 'Location', 'best');
    end
    
    % Restore warnings
    warning('on', 'all');
end