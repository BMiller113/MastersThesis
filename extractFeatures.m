function features = extractFeatures(audioFiles)
    % Extract log-mel filterbank features from audio files
    numBands = 40;         % Number of mel bands
    targetFrames = 32;     % 23 left + 8 right + current frame (see paper)
    frameDuration = 0.025; % 25ms frames
    overlapDuration = 0.01; % 10ms hop size
    
    % Initialize feature array
    numFiles = length(audioFiles);
    features = zeros(numBands, targetFrames, 1, numFiles, 'single');
    
    % Create mel filterbank
    melFilterbank = [];
    
    for i = 1:numFiles
        try
            % Read and normalize audio
            [audioIn, fs] = audioread(audioFiles{i});
            audioIn = audioIn / max(abs(audioIn(:))); % Peak normalization
            
            % Initialize mel filterbank if not done
            if isempty(melFilterbank)
                frameLength = round(frameDuration * fs);
                melFilterbank = designAuditoryFilterBank(fs, ...
                    'NumBands', numBands, ...
                    'FFTLength', frameLength);
            end
            
            % STFT
            frameLength = round(frameDuration * fs);
            overlapLength = round(overlapDuration * fs);
            [s, ~, ~] = spectrogram(audioIn, hamming(frameLength), ...
                overlapLength, frameLength, fs);
            
            % Mel spectrogram
            melSpectrogram = melFilterbank * abs(s);
            
            % Log compression with floor
            logMelSpectrogram = log10(max(melSpectrogram, 1e-6)); % log(0) bad
            
            % Normalize spectrogram (big deal dont forget in future)
            logMelSpectrogram = (logMelSpectrogram - mean(logMelSpectrogram(:))) ./ ...
                (std(logMelSpectrogram(:)) + 1e-6); % Add epsilon to avoid /0
            
            % Dynamic frame selection (center)
            numAvailableFrames = size(logMelSpectrogram, 2);
            if numAvailableFrames >= targetFrames
                center = round(numAvailableFrames/2);
                startFrame = max(1, center - 15);
                endFrame = min(numAvailableFrames, startFrame + targetFrames - 1);
                logMelSpectrogram = logMelSpectrogram(:, startFrame:endFrame);
            else
                % Pad with zeros
                logMelSpectrogram = [logMelSpectrogram, ...
                    zeros(numBands, targetFrames - numAvailableFrames)];
            end
            
            % Ensure exact frame count
            features(:, :, 1, i) = logMelSpectrogram(:, 1:targetFrames);
            
        catch ME
            warning('Error processing file %s: %s', audioFiles{i}, ME.message);
            features(:, :, 1, i) = zeros(numBands, targetFrames);
        end
    end
    
    % Global normalization
    features = (features - mean(features(:))) / std(features(:));
end