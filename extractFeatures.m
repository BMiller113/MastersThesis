function features = extractFeatures(audioFiles)
    % Extract log-mel filterbank features from audio files
    numFiles = length(audioFiles);
    numBands = 40; % Number of mel bands
    targetFrames = 100;

    % Initialize the 4D array for features
    features = zeros(numBands, targetFrames, 1, numFiles); % [height, width, channels, numObservations]

    % Define parameters for feature extraction
    frameDuration = 0.025;  %ms
    overlapDuration = 0.01;

    for i = 1:numFiles
        [audioIn, fs] = audioread(audioFiles{i});
        audioIn = audioIn / max(abs(audioIn));

        % Compute frame length and overlap in samples
        frameLength = round(frameDuration * fs);
        overlapLength = round(overlapDuration * fs);

        % Compute the Short-Time Fourier Transform (STFT)
        [s, f, t] = spectrogram(audioIn, hamming(frameLength), overlapLength, frameLength, fs);

        % Compute the magnitude spectrogram
        spectrogramMagnitude = abs(s);

        % Compute the mel filterbank
        melFilterbank = designAuditoryFilterBank(fs, 'NumBands', numBands, 'FFTLength', frameLength);

        % Apply the mel filterbank to the spectrogram
        melSpectrogram = melFilterbank * spectrogramMagnitude;

        % Convert to log scale
        logMelSpectrogram = log10(melSpectrogram + eps);

        % Truncate or pad the spectrogram to the target number of frames
        % Truncation very important
        [numBands, numFrames] = size(logMelSpectrogram);
        if numFrames < targetFrames
            % Pad with zeros
            logMelSpectrogram = [logMelSpectrogram, zeros(numBands, targetFrames - numFrames)];
        elseif numFrames > targetFrames
            % Truncate
            logMelSpectrogram = logMelSpectrogram(:, 1:targetFrames);
        end

        % Store the features
        features(:, :, 1, i) = logMelSpectrogram; % [numBands, numFrames, 1, numFiles]
    end
end