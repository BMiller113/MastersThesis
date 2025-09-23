function [XTrain, YTrain, XTest, YTest] = loadGenderSplitData(genderMap, genderFilter, useMelFilter, melMode, cfg)
    if nargin < 5, cfg = []; end

    [trainingFiles, trainingLabels, testingFiles, testingLabels] = loadAudioData();

    if ~strcmp(genderFilter, 'all')
        trainingKeep = filterByGender(trainingFiles, genderMap, genderFilter);
        testingKeep  = filterByGender(testingFiles,  genderMap, genderFilter);
        trainingFiles  = trainingFiles(trainingKeep);
        trainingLabels = trainingLabels(trainingKeep);
        testingFiles   = testingFiles(testingKeep);
        testingLabels  = testingLabels(testingKeep);
    end

    if useMelFilter
        [XTrain, validTrain] = extractFeatures(trainingFiles, genderFilter, melMode, cfg);
        [XTest,  validTest ] = extractFeatures(testingFiles,  genderFilter, melMode, cfg);
    else
        % Non-mel specialization path still uses the config for bands/frames
        [XTrain, validTrain] = extractFeatures(trainingFiles, 'all', 'default', cfg);
        [XTest,  validTest ] = extractFeatures(testingFiles,  'all', 'default', cfg);
    end

    % Slice labels by validity
    YTrain = trainingLabels(validTrain);
    YTest  = testingLabels(validTest);
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
