function gender = extract_gender(audioFile)
    % Estimate gender based on median pitch using pitch detection
    try
        [audio, fs] = audioread(audioFile);
        audio = audio / max(abs(audio));  % Normalize

        pitchVals = pitch(audio, fs, 'Range', [50, 400]);  % Human pitch range
        medPitch = median(pitchVals, 'omitnan');

        if isnan(medPitch)
            gender = 'unknown';
        elseif medPitch > 165
            gender = 'female';
        else
            gender = 'male';
        end
    catch
        warning('Pitch estimation failed for %s. Marking as unknown.', audioFile);
        gender = 'unknown';
    end
end
