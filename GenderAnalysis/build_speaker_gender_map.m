% Run this to build speaker map
function genderMap = build_speaker_gender_map(datasetPath)
    % Recursively find .wav files
    files = dir(fullfile(datasetPath, '**', '*.wav'));
    genderMap = containers.Map();

    processedSpeakers = containers.Map();

    for k = 1:length(files)
        filePath = fullfile(files(k).folder, files(k).name);
        [~, filename, ~] = fileparts(filePath);

        % Get speaker ID
        underscoreIdx = strfind(filename, '_');
        if isempty(underscoreIdx)
            continue;
        end
        speakerID = filename(1:underscoreIdx(1)-1);

        if isKey(processedSpeakers, speakerID)
            continue;  % Speaker already processed
        end

        gender = extract_gender(filePath);
        if ~strcmp(gender, 'unknown')
            genderMap(speakerID) = gender;
        end

        processedSpeakers(speakerID) = true;
    end

    fprintf('Total speakers processed: %d\n', length(genderMap));
end
