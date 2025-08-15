function [XTrain, YTrain, XTest, YTest] = loadGenderSplitData(genderMap, genderFilter, useMelFilter, melMode)

    if nargin < 3 || isempty(useMelFilter), useMelFilter = false; end
    if nargin < 4 || isempty(melMode),      melMode      = 'default'; end

    % Load full lists
    [trainingFiles, trainingLabels, testingFiles, testingLabels] = loadAudioData();

    % Filter by gender as requested
    if ~strcmpi(genderFilter, 'all')
        trainingKeep = filterByGender(trainingFiles, genderMap, genderFilter);
        testingKeep  = filterByGender(testingFiles,  genderMap, genderFilter);

        trainingFiles  = trainingFiles(trainingKeep);
        trainingLabels = trainingLabels(trainingKeep);

        testingFiles   = testingFiles(testingKeep);
        testingLabels  = testingLabels(testingKeep);
    end

    % Choose mel mode to use
    if useMelFilter
        melModeEff = melMode;
    else
        melModeEff = 'default';
    end
    if isstring(melModeEff), melModeEff = char(melModeEff); end
    if isstring(genderFilter), genderFilter = char(genderFilter); end

    % Extract features (returns validIdx masks)
    [XTrain, validTrain] = extractFeatures(trainingFiles, genderFilter, melModeEff);
    [XTest,  validTest ] = extractFeatures(testingFiles,  genderFilter, melModeEff);

    % Slice both features and labels by validIdx
    XTrain = XTrain(:,:,:,validTrain);
    XTest  = XTest(:,:,:,validTest);


    YTrain = trainingLabels(validTrain);
    YTest  = testingLabels(validTest);

    %  Clean up labels: make categorical and drop unused classes
    %   %8/15 phantom class fix
    if ~iscategorical(YTrain), YTrain = categorical(YTrain); end  
    if ~iscategorical(YTest),  YTest  = categorical(YTest);  end
    YTrain = removecats(YTrain);
    YTest  = removecats(YTest);

    % Sanity checks
    assert(size(XTrain,4) == numel(YTrain), ...
        'loadGenderSplitData:TrainCountMismatch: XTrain has %d items, YTrain has %d.', ...
        size(XTrain,4), numel(YTrain));

    assert(size(XTest,4) == numel(YTest), ...
        'loadGenderSplitData:TestCountMismatch: XTest has %d items, YTest has %d.', ...
        size(XTest,4), numel(YTest));
end

function keepMask = filterByGender(fileList, genderMap, targetGender)
    keepMask = false(size(fileList));
    for i = 1:numel(fileList)
        [~, fileName, ~] = fileparts(fileList{i});
        underscoreIdx = strfind(fileName, '_');
        if isempty(underscoreIdx), continue; end
        speakerID = fileName(1:underscoreIdx(1)-1);
        if isKey(genderMap, speakerID)
            keepMask(i) = strcmpi(genderMap(speakerID), targetGender);
        end
    end
end
