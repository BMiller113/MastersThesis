function accuracy = evaluateModel(net, XTest, YTest)
    % Predict using the trained network
    YPred = classify(net, XTest);
    
    % Calculate accuracy
    correctPredictions = sum(YPred == YTest);
    totalSamples = numel(YTest);
    accuracy = (correctPredictions / totalSamples) * 100;
    
    % Display detailed breakdown
    disp(' ');
    disp('=== Evaluation Details ===');
    disp(['Correct predictions: ', num2str(correctPredictions)]);
    disp(['Total test samples: ', num2str(totalSamples)]);
    disp(['Accuracy: ', num2str(accuracy, '%.2f'), '%']);
    disp('=========================');
end