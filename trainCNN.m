function net = trainCNN(XTrain, YTrain, layers)
    % 20% holdout without stratification
    n = numel(YTrain);
    holdoutIdx = false(n, 1);
    rng(42);  % reproducible
    holdoutIdx(randperm(n, round(0.2 * n))) = true;

    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 128, ...
        'ValidationData', {XTrain(:,:,:,holdoutIdx), YTrain(holdoutIdx)}, ...
        'ValidationFrequency', 30, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'L2Regularization', 0.001, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    net = trainNetwork(XTrain(:,:,:,~holdoutIdx), YTrain(~holdoutIdx), layers, options);
end