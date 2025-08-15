function runGenderTrainingEvaluation(genderFilter)
    % Load gender map
    load('speakerGenderMap.mat', 'genderMap');

    % Load gender-filtered or full data
    fprintf('Loading data with gender filter: %s\n', genderFilter);
    [XTrain, YTrain, XTest, YTest] = loadGenderSplitData(genderMap, genderFilter);

    % Define architecture
    archType = 'trad-fpool3';
    layers = defineCNNArchitecture(numel(categories(YTrain)), archType);

    % Train model and evaluate
    fprintf('Training model...\n');
    net = trainCNN(XTrain, YTrain, layers);
    fprintf('Evaluating model...\n');
    [accuracy, FR, FA, rocInfo] = evaluateModel(net, XTest, YTest);
    fprintf('\n=== Results (%s set) ===\n', genderFilter);
    fprintf('Accuracy: %.2f%%\n', accuracy);
    fprintf('False Reject Rate: %.2f%%\n', FR);
    fprintf('False Alarm Rate: %.2f%%\n', FA);
end
