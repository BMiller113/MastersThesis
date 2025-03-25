function [audioFiles, labels, testingFiles, validationFiles] = loadAudioData()
    dataPath = 'C:\Users\bjren\MATLAB\Projects\KeywordSpottingThesis\Data\Kaggle_ GoogleSpeechCommandsV2';
    % Safetys
    if ~exist(dataPath, 'dir')
        error('Dataset folder not found: %s', dataPath);
    end
    % Load Datasets
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

    % Get a list of all .wav files in the dataset
    audioFiles = dir(fullfile(dataPath, '**/*.wav')); % Recursively search for .wav files
    audioFiles = fullfile({audioFiles.folder}, {audioFiles.name});

    % Extract labels from filenames
    labels = cell(size(audioFiles));
    isTesting = false(size(audioFiles));
    isValidation = false(size(audioFiles));

    for i = 1:length(audioFiles)
        [folder, filename, ~] = fileparts(audioFiles{i});
        [~, label, ~] = fileparts(folder);
        labels{i} = label;

        % Construct the relative path for comparison
        relativePath = strrep(audioFiles{i}, [dataPath, filesep], ''); % Remove base path
        relativePath = strrep(relativePath, '\', '/'); % Ensure consistent seperators

        % Check if the file is testing/validation/other(training)
        isTesting(i) = ismember(relativePath, testingList);
        isValidation(i) = ismember(relativePath, validationList);
    end

    labels = categorical(labels);

    % Split into training, testing, and validation sets
    trainingFiles = audioFiles(~isTesting & ~isValidation);
    testingFiles = audioFiles(isTesting);
    validationFiles = audioFiles(isValidation);
    disp(['Training files: ', num2str(length(trainingFiles))]);
    disp(['Testing files: ', num2str(length(testingFiles))]);
    disp(['Validation files: ', num2str(length(validationFiles))]);
end

function lines = readLines(filename)
    % Read all lines from a text file
    fileID = fopen(filename, 'r');
    if fileID == -1
        error('Could not open file: %s', filename);
    end
    lines = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);
    lines = lines{1};

    % Remove any whitespaces (shouldnt be an issue but just in case)
    lines = strtrim(lines);
end