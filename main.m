% Main script for CNN-based Keyword Spotting with Gender Control
% If no speaker map, run build_speaker_gender_map

% Load gender map
load('speakerGenderMap.mat', 'genderMap');

% Configurations to test
genderModes = {'none', 'filter', 'filter+mel'};     % Preprocessing levels
melModes    = {'default', 'narrow', 'wide'};        % Filterbank styles

% Loop through all combinations
for g = 1:length(genderModes)
    for m = 1:length(melModes)

        genderMode = genderModes{g};    % 'none' / 'filter' / 'filter+mel'
        melMode = melModes{m};          % 'default' / 'narrow' / 'wide'

        fprintf('\n=== GENDER MODE: %s | MEL MODE: %s ===\n', upper(genderMode), upper(melMode));

        % Determine which filter to use
        switch genderMode
            case 'none'
                filterGender = 'all';
                useMelFilter = false;
            case 'filter'
                filterGender = 'male';  % change this to 'female' to test other group
                useMelFilter = false;
            case 'filter+mel'
                filterGender = 'male';  % change this to 'female' to test other group
                useMelFilter = true;
            otherwise
                error('Unknown genderMode: %s', genderMode);
        end

        % === Load data ===
        [XTrain, YTrain, XTest, YTest] = loadGenderSplitData(genderMap, filterGender, useMelFilter, melMode);

        if numel(categories(YTrain)) < 2
            warning('Training set missing a class. Skipping...');
            continue;
        end

        % === Define model ===
        archType = 'tpool2'; % try: 'one-fstride4', 'tpool2', 'trad-fpool3'
        layers = defineCNNArchitecture(numel(categories(YTrain)), archType);
        net = trainCNN(XTrain, YTrain, layers);

        % === Evaluate ===
        [accuracy, FR, FA, rocInfo] = evaluateModel(net, XTest, YTest);

        % === Save ===
        tag = sprintf('%s_%s', genderMode, melMode);
        save(['model_' tag '.mat'], 'net', 'archType');
        results = struct('GenderMode', genderMode, 'MelMode', melMode, ...
                         'Accuracy', accuracy, 'FR', FR, 'FA', FA, ...
                         'Timestamp', datetime());
        save(['results_' tag '.mat'], 'results', 'rocInfo');

        % === Output ===
        fprintf('Accuracy: %.2f%% | FR: %.2f%% | FA: %.2f%%\n', accuracy, FR, FA);
    end
end
