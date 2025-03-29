function [accuracy, FR, FA] = evaluateModel(net, XTest, YTest)
    % Turn off warnings during evaluation
    warning('off', 'all');
    
    % Get predictions
    YPred = classify(net, XTest);
    scores = predict(net, XTest);
    accuracy = mean(YPred == YTest) * 100;
    
    classNames = categories(YTest);
    positiveClass = classNames{1};
    
    % Calculate FR/FA @ 0.5 threshold
    isPositive = (YTest == positiveClass);
    isNegative = ~isPositive;
    predPositive = scores(:,1) >= 0.5;
    
    FR = sum(~predPositive & isPositive) / sum(isPositive) * 100;
    FA = sum(predPositive & isNegative) / sum(isNegative) * 100;
    
    % Out
    fprintf('\n=== Final Evaluation ===\n');
    fprintf('Accuracy:       %.2f%%\n', accuracy);
    fprintf('False Reject:   %.2f%%\n', FR);
    fprintf('False Alarm:    %.2f%%\n\n', FA);
    
    % Restore warnings
    warning('on', 'all');
end