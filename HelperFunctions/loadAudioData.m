function [trainingFiles, trainingLabels, testingFiles, testingLabels] = loadAudioData()
    % Load dataset path from config
    cfg = kws_config();
    dataPath = cfg.paths.datasetRoot;

    % Check if data exists
    if ~exist(dataPath, 'dir')
        error('Dataset folder not found: %s', dataPath);
    end

    % Load testing and validation file lists
    testingListPath    = fullfile(dataPath, 'testing_list.txt');
    validationListPath = fullfile(dataPath, 'validation_list.txt');

    if ~exist(testingListPath, 'file')
        error('Testing list not found: %s', testingListPath);
    end
    if ~exist(validationListPath, 'file')
        error('Validation list not found: %s', validationListPath);
    end

    testingList    = readLines(testingListPath);
    validationList = readLines(validationListPath);

    % Get all .wav files (recursively)
    audioFilesStruct = dir(fullfile(dataPath, '**', '*.wav'));
    audioFiles = fullfile({audioFilesStruct.folder}, {audioFilesStruct.name})';

    % Extract labels
    numFiles     = numel(audioFiles);
    labels       = cell(numFiles, 1);
    isTesting    = false(numFiles, 1);
    isValidation = false(numFiles, 1);

    dprefix = [dataPath, filesep];
    for i = 1:numFiles
        [folder, ~, ~] = fileparts(audioFiles{i});
        [~, label] = fileparts(folder);
        labels{i} = label;

        % Dataset-relative path (Google Speech Commands Kaggle V2 8/15) using forward slashes 
        relativePath = strrep(audioFiles{i}, dprefix, '');
        relativePath = strrep(relativePath, '\', '/');

        isTesting(i)    = ismember(relativePath, testingList);
        isValidation(i) = ismember(relativePath, validationList);
    end

    % Convert labels to categorical
    labels = categorical(labels);

    % Split into sets
    trainingMask   = ~isTesting & ~isValidation;
    trainingFiles  = audioFiles(trainingMask);
    testingFiles   = audioFiles(isTesting);
    validationFiles = audioFiles(isValidation); %#ok<NASGU>

    trainingLabels = labels(trainingMask);
    testingLabels  = labels(isTesting);

    disp(['Training files: ', num2str(numel(trainingFiles))]);
    disp(['Testing files: ',  num2str(numel(testingFiles))]);
    disp(['Validation files: ', num2str(numel(find(isValidation)))]);

end

function lines = readLines(filename)
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end
    C = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    lines = strtrim(C{1});
end
