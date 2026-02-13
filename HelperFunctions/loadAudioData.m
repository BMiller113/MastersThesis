function [trainingFiles, trainingLabels, testingFiles, testingLabels] = loadAudioData(cfg)
% loadAudioData
% Loads Google Speech Commands using official testing_list.txt/validation_list.txt.
% Uses cfg.dataset.version to choose cfg.paths.datasetRootV1 or cfg.paths.datasetRootV2.
%
% Outputs:
%   trainingFiles, trainingLabels, testingFiles, testingLabels

    if nargin < 1 || isempty(cfg)
        cfg = kws_config();
    end

    % Resolve dataset root
    v = 'v2';
    if isfield(cfg,'dataset') && isfield(cfg.dataset,'version') && ~isempty(cfg.dataset.version)
        v = lower(string(cfg.dataset.version));
    end

    switch v
        case "v1"
            dataPath = cfg.paths.datasetRootV1;
        otherwise
            dataPath = cfg.paths.datasetRootV2;
    end

    if ~exist(dataPath, 'dir')
        error('Dataset folder not found: %s', dataPath);
    end

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

    % All wav files
    audioFilesStruct = dir(fullfile(dataPath, '**', '*.wav'));
    audioFiles = fullfile({audioFilesStruct.folder}, {audioFilesStruct.name})';
    if isempty(audioFiles)
        error('No .wav files found under: %s', dataPath);
    end

    % Extract labels from parent folder
    numFiles     = numel(audioFiles);
    labels       = cell(numFiles, 1);
    isTesting    = false(numFiles, 1);
    isValidation = false(numFiles, 1);

    dprefix = [dataPath, filesep];

    for i = 1:numFiles
        [folder, ~, ~] = fileparts(audioFiles{i});
        [~, label] = fileparts(folder);
        labels{i} = label;

        rel = strrep(audioFiles{i}, dprefix, '');
        rel = strrep(rel, '\', '/'); % lists use forward slashes

        isTesting(i)    = ismember(rel, testingList);
        isValidation(i) = ismember(rel, validationList);
    end

    labels = categorical(labels);

    trainingMask    = ~isTesting & ~isValidation;
    testingMask     = isTesting;

    trainingFiles   = audioFiles(trainingMask);
    testingFiles    = audioFiles(testingMask);

    trainingLabels  = labels(trainingMask);
    testingLabels   = labels(testingMask);

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
