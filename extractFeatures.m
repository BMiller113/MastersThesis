function [features, validIdx] = extractFeatures(audioFiles)
    % Parameters from paper
    numBands = 40;          % Number of mel bands
    targetFrames = 32;      % Number of time frames
    frameDuration = 0.025;  % 25ms frames
    overlapDuration = 0.01; % 10ms hop size

    % Initialize feature array
    numFiles = length(audioFiles);
    features = zeros(numBands, targetFrames, 1, numFiles, 'single');
    validIdx = true(1, numFiles);  % <-- Track valid features

    % Process each file
    for i = 1:numFiles
        try
            [audioIn, fs] = audioread(audioFiles{i});
            audioIn = audioIn / max(abs(audioIn(:)));

            frameLength = round(frameDuration * fs);
            overlapLength = round(overlapDuration * fs);
            [s, ~, ~] = spectrogram(audioIn, hamming(frameLength), overlapLength, frameLength, fs);

            melFilterbank = designAuditoryFilterBank(fs, 'NumBands', numBands, 'FFTLength', frameLength);
            melSpectrogram = melFilterbank * abs(s);

            logMelSpectrogram = log10(melSpectrogram + eps);

            [~, numFrames] = size(logMelSpectrogram);
            if numFrames < targetFrames
                logMelSpectrogram = [logMelSpectrogram, zeros(numBands, targetFrames - numFrames)];
            elseif numFrames > targetFrames
                startFrame = floor((numFrames - targetFrames)/2) + 1;
                logMelSpectrogram = logMelSpectrogram(:, startFrame:startFrame+targetFrames-1);
            end

            features(:, :, 1, i) = logMelSpectrogram;

        catch ME
            warning('Error processing %s: %s', audioFiles{i}, ME.message);
            features(:, :, 1, i) = zeros(numBands, targetFrames);
            validIdx(i) = false;  % <-- Mark as invalid
        end
    end

    % Global normalization
    features = (features - mean(features(:))) / std(features(:));
end
