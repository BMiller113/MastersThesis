function make_streams_long_extraNeg()
    cfg = kws_config('sainath14');

    numStreams   = 60;
    streamLenSec = 180;

    frameMs      = cfg.features.frameMs;   % 25
    hopMs        = cfg.features.hopMs;     % 10
    spanMs       = frameMs + hopMs*(23+8);
    hopWinMs     = 10;

    [~, ~, testFiles, testLabels] = loadAudioData();

    baseDir = fullfile(pwd, 'streams_long_extraNeg');
    if ~exist(baseDir,'dir'), mkdir(baseDir); end

    % -------- CLEAN: fewer keywords, tighter labels --------
    cfgC = cfg;
    cfgC.streaming.numStreams     = numStreams;
    cfgC.streaming.streamLenSec   = streamLenSec;
    cfgC.streaming.winSpanMs      = spanMs;
    cfgC.streaming.hopWinMs       = hopWinMs;
    cfgC.streaming.noiseSNRdB     = [];
    cfgC.streaming.bgGain         = 0.0;
    cfgC.streaming.keywordsPerMin = 5;     % <-- was 15
    cfgC.streaming.minGapSec      = 0.5;   % a little more space
    % tighten label tol to be closer to Sainath (~100 ms)
    cfgC.sainath.labelTolMs       = 100;

    wavCleanDir = fullfile(baseDir, 'clean_wav');
    if ~exist(wavCleanDir,'dir'), mkdir(wavCleanDir); end

    streams_clean = makeStreamingCorpus(cfgC, testFiles, testLabels, wavCleanDir);
    streams_clean = addStreamingWindows(streams_clean, cfgC);

    for s = 1:numel(streams_clean)
        if ~isfile(streams_clean(s).wavPath)
            streams_clean(s).wavPath = fullfile(wavCleanDir, sprintf('stream_%02d.wav', s));
        end
    end

    save(fullfile(baseDir, 'streams_long_extraNeg_clean.mat'), 'streams_clean', '-v7.3');

    % -------- NOISY: can stay at 10 dB, but we can also reduce keyword rate --------
    cfgN = cfg;
    cfgN.streaming.numStreams     = numStreams;
    cfgN.streaming.streamLenSec   = streamLenSec;
    cfgN.streaming.winSpanMs      = spanMs;
    cfgN.streaming.hopWinMs       = hopWinMs;
    cfgN.streaming.noiseSNRdB     = 10;
    cfgN.streaming.bgGain         = 0.3;
    cfgN.streaming.keywordsPerMin = 5;     % match clean to get more negatives
    cfgN.sainath.labelTolMs       = 100;

    wavNoisyDir = fullfile(baseDir, 'noisy_wav');
    if ~exist(wavNoisyDir,'dir'), mkdir(wavNoisyDir); end

    streams_noisy = makeStreamingCorpus(cfgN, testFiles, testLabels, wavNoisyDir);
    streams_noisy = addStreamingWindows(streams_noisy, cfgN);

    for s = 1:numel(streams_noisy)
        if ~isfile(streams_noisy(s).wavPath)
            streams_noisy(s).wavPath = fullfile(wavNoisyDir, sprintf('stream_%02d.wav', s));
        end
    end

    save(fullfile(baseDir, 'streams_long_extraNeg_noisy.mat'), 'streams_noisy', '-v7.3');
end
