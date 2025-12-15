function make_streams_10h_bg()
% Build big streaming sets (~10 hours/condition), background-heavy:
%  - very low keyword density (1–2 per minute)
%  - tighter labeling tol (50–100 ms)
%  - noisy set varies SNR per stream (5/10/15 dB) to diversify negatives
%
% Saves:
%   ./streams_10h_bg/streams_10h_bg_clean.mat
%   ./streams_10h_bg/streams_10h_bg_noisy.mat

    cfg = kws_config('sainath14');

    numStreams   = 100;   % 100 x 360s = 10 hours
    streamLenSec = 360;   % 6 minutes each

    frameMs      = cfg.features.frameMs;    % 25
    hopMs        = cfg.features.hopMs;      % 10
    spanMs       = frameMs + hopMs*(23+8);  % ~335 ms
    hopWinMs     = 10;                      % 10 ms decisions

    [~, ~, testFiles, testLabels] = loadAudioData();

    baseDir = fullfile(pwd, 'streams_10h_bg');
    if ~exist(baseDir,'dir'), mkdir(baseDir); end

    %% -------- CLEAN: very few keywords, tighter tol --------
    cfgC = cfg;
    cfgC.streaming.numStreams     = numStreams;
    cfgC.streaming.streamLenSec   = streamLenSec;
    cfgC.streaming.winSpanMs      = spanMs;
    cfgC.streaming.hopWinMs       = hopWinMs;
    cfgC.streaming.keywordsPerMin = 2;      % << was 15
    cfgC.streaming.minGapSec      = 1.2;    % more spacing
    cfgC.streaming.noiseSNRdB     = [];
    cfgC.streaming.bgGain         = 0.0;
    cfgC.sainath.labelTolMs       = 100;    % try 50–100

    wavCleanDir = fullfile(baseDir, 'clean_wav');
    if ~exist(wavCleanDir,'dir'), mkdir(wavCleanDir); end

    streams_clean = makeStreamingCorpus(cfgC, testFiles, testLabels, wavCleanDir);
    streams_clean = addStreamingWindows(streams_clean, cfgC);

    % fix wavPath + per-stream stats
    totalPos = 0; totalNeg = 0; totalWin = 0;
    for s = 1:numel(streams_clean)
        if ~isfile(streams_clean(s).wavPath)
            streams_clean(s).wavPath = fullfile(wavCleanDir, sprintf('stream_%02d.wav', s));
        end
        labs = streams_clean(s).winLabels;
        totalWin = totalWin + numel(labs);
        totalPos = totalPos + sum(labs ~= "_neg_");
        totalNeg = totalNeg + sum(labs == "_neg_");
    end
    fprintf('CLEAN 10h: windows=%d, pos=%d, neg=%d (neg:pos=%.1f:1)\n', ...
        totalWin, totalPos, totalNeg, totalNeg/max(1,totalPos));

    save(fullfile(baseDir, 'streams_10h_bg_clean.mat'), 'streams_clean', '-v7.3');
    fprintf('make_streams_10h_bg: wrote CLEAN 10h -> %s\n', ...
        fullfile(baseDir, 'streams_10h_bg_clean.mat'));

    %% -------- NOISY: same, but vary SNR to diversify negatives --------
    wavNoisyDir = fullfile(baseDir, 'noisy_wav');
    if ~exist(wavNoisyDir,'dir'), mkdir(wavNoisyDir); end

    snrChoices = [5 10 15];

    % Build first to get struct shape
    cfgN1 = cfg;
    cfgN1.streaming.numStreams     = 1;
    cfgN1.streaming.streamLenSec   = streamLenSec;
    cfgN1.streaming.winSpanMs      = spanMs;
    cfgN1.streaming.hopWinMs       = hopWinMs;
    cfgN1.streaming.keywordsPerMin = 2;
    cfgN1.streaming.minGapSec      = 1.2;
    cfgN1.streaming.bgGain         = 0.3;
    cfgN1.streaming.noiseSNRdB     = snrChoices(randi(numel(snrChoices)));
    cfgN1.sainath.labelTolMs       = 100;

    tmp1 = makeStreamingCorpus(cfgN1, testFiles, testLabels, wavNoisyDir);
    tmp1 = addStreamingWindows(tmp1, cfgN1);
    if ~isfile(tmp1(1).wavPath)
        tmp1(1).wavPath = fullfile(wavNoisyDir, sprintf('stream_%02d.wav', 1));
    end
    streams_noisy = repmat(tmp1(1), numStreams, 1);

    % Fill remaining streams with varied SNR
    for s = 2:numStreams
        cfgNs = cfg;
        cfgNs.streaming.numStreams     = 1;
        cfgNs.streaming.streamLenSec   = streamLenSec;
        cfgNs.streaming.winSpanMs      = spanMs;
        cfgNs.streaming.hopWinMs       = hopWinMs;
        cfgNs.streaming.keywordsPerMin = 2;
        cfgNs.streaming.minGapSec      = 1.2;
        cfgNs.streaming.bgGain         = 0.3;
        cfgNs.streaming.noiseSNRdB     = snrChoices(randi(numel(snrChoices)));
        cfgNs.sainath.labelTolMs       = 100;

        tmp = makeStreamingCorpus(cfgNs, testFiles, testLabels, wavNoisyDir);
        tmp = addStreamingWindows(tmp, cfgNs);
        if ~isfile(tmp(1).wavPath)
            tmp(1).wavPath = fullfile(wavNoisyDir, sprintf('stream_%02d.wav', s));
        end
        streams_noisy(s) = tmp(1);
    end

    % stats
    totalPos = 0; totalNeg = 0; totalWin = 0;
    for s = 1:numel(streams_noisy)
        labs = streams_noisy(s).winLabels;
        totalWin = totalWin + numel(labs);
        totalPos = totalPos + sum(labs ~= "_neg_");
        totalNeg = totalNeg + sum(labs == "_neg_");
    end
    fprintf('NOISY 10h: windows=%d, pos=%d, neg=%d (neg:pos=%.1f:1)\n', ...
        totalWin, totalPos, totalNeg, totalNeg/max(1,totalPos));

    save(fullfile(baseDir, 'streams_10h_bg_noisy.mat'), 'streams_noisy', '-v7.3');
    fprintf('make_streams_10h_bg: wrote NOISY 10h -> %s\n', ...
        fullfile(baseDir, 'streams_10h_bg_noisy.mat'));
end
