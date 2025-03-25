function [XTrain, YTrain, XTest, YTest] = loadAndPreprocessData()
    [audioFiles, labels] = loadAudioData();
    features = extractFeatures(audioFiles); % Melbank feature extraction
    [XTrain, YTrain, XTest, YTest] = splitData(features, labels);
end