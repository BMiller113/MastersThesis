function [trainingFiles, trainingLabels, testingFiles, testingLabels] = loadAudioData()
    % Define dataset path
    dataPath = 'C:\Users\bjren\MATLAB\Projects\KeywordSpottingThesis\Data\Kaggle_ GoogleSpeechCommandsV2';

    % Check if data exists
    if ~exist(dataPath, 'dir')
        error('Dataset folder not found: %s', dataPath);
    end

    % Load testing and validation file lists
    testingListPath = fullfile(dataPath, 'testing_list.txt');
    validationListPath = fullfile(dataPath, 'validation_list.txt');

    if ~exist(testingListPath, 'file')
        error('Testing list not found: %s', testingListPath);
    end
    if ~exist(validationListPath, 'file')
        error('Validation list not found: %s', validationListPath);
    end

    testingList = readLines(testingListPath);
    validationList = readLines(validationListPath);

    % Get all .wav files recursively
    audioFilesStruct = dir(fullfile(dataPath, '**/*.wav'));
    audioFiles = fullfile({audioFilesStruct.folder}, {audioFilesStruct.name})';

    % Extract labels and partition info
    numFiles = numel(audioFiles);
    labels = cell(numFiles, 1);
    isTesting = false(numFiles, 1);
    isValidation = false(numFiles, 1);

    for i = 1:numFiles
        [folder, ~, ~] = fileparts(audioFiles{i});
        [~, label] = fileparts(folder);
        labels{i} = label;

        relativePath = strrep(audioFiles{i}, [dataPath, filesep], '');
        relativePath = strrep(relativePath, '\', '/'); % standardize separators

        isTesting(i) = ismember(relativePath, testingList);
        isValidation(i) = ismember(relativePath, validationList);
    end

    % Convert labels to categorical
    labels = categorical(labels);

    % Split into sets
    trainingFiles = audioFiles(~isTesting & ~isValidation);
    testingFiles = audioFiles(isTesting);
    validationFiles = audioFiles(isValidation);

    trainingLabels = labels(~isTesting & ~isValidation);
    testingLabels = labels(isTesting);
    % validationLabels = labels(isValidation); % optional if needed

    disp(['Training files: ', num2str(length(trainingFiles))]);
    disp(['Testing files: ', num2str(length(testingFiles))]);
    disp(['Validation files: ', num2str(length(validationFiles))]);

    % Optional: return validation too by updating output vars

    % Placeholder: Optional noise augmentation (disabled here)
    % if ~exist('testingFiles', 'var')
    %     ... augmentation logic here ...
    % end
end

function lines = readLines(filename)
    fileID = fopen(filename, 'r');
    if fileID == -1
        error('Could not open file: %s', filename);
    end
    lines = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);
    lines = strtrim(lines{1});
end
