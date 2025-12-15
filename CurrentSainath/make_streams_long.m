function make_streams_long()
% make_streams_long
% Build longer Sainath-style streaming sets with sparse keywords and a
% small CLEAN background bed to de-flat the negative posterior shelf.
%
%   - clean : 60 streams × 180 s  (≈ 3 hours total), quiet room-tone bg
%   - noisy : 60 streams × 180 s  (≈ 3 hours total), ~10 dB SNR
%
% Sainath geometry: 25ms/10ms, 23L+1+8R -> ~335 ms span, 10 ms decisions
%
% Saves to:
%   <pwd>\streams_long\streams_long_clean.mat
%   <pwd>\streams_long\streams_long_noisy.mat
%
% Uses your existing helpers:
%   kws_config('sainath14'), loadAudioData(), makeStreamingCorpus(), addStreamingWindows()

    cfg = kws_config('sainath14');

    % ---- length / count (adjust if you want even longer) ----
    numStreams   = 60;     % e.g., bump to 120 for ~6 hours/cond
    streamLenSec = 180;    % 3 minutes per stream

    % ---- Sainath geometry ----
    frameMs      = cfg.features.frameMs;   % 25
    hopMs        = cfg.features.hopMs;     % 10
    spanMs       = frameMs + hopMs*(23+8); % 335
    hopWinMs     = 10;                     % decisions every 10 ms

    % ---- Sainath-ish labeling tolerance (tighter than 200) ----
    labelTolMs   = 100;

    % ---- Sparse KW rate to lower P% (de-bias the prior) ----
    keywordsPerMin = 1.5;   % 1–2 per minute is good
    minGapSec      = 1.2;   % leave breathing room

    % ---- Background levels ----
    cleanBgGain    = 0.03;  % tiny “room tone” so negatives vary
    noisySNRdB     = 10;    % light noise for noisy condition
    noisyBgGain    = 0.30;  % fallback if SNR mixing not used

    % ---- dataset files ----
    [~, ~, testFiles, testLabels] = loadAudioData();

    baseDir = fullfile(pwd, 'streams_long');
    if ~exist(baseDir,'dir'), mkdir(baseDir); end

    % ---------------- CLEAN ----------------
    cfgC = cfg;
    cfgC.streaming.numStreams     = numStreams;
    cfgC.streaming.streamLenSec   = streamLenSec;
    cfgC.streaming.winSpanMs      = spanMs;
    cfgC.streaming.hopWinMs       = hopWinMs;
    cfgC.streaming.keywordsPerMin = keywordsPerMin;
    cfgC.streaming.minGapSec      = minGapSec;
    cfgC.streaming.noiseSNRdB     = [];           % marks CLEAN (use bgGain)
    cfgC.streaming.bgGain         = cleanBgGain;  % << key: add quiet bed
    cfgC.sainath.labelTolMs       = labelTolMs;

    wavCleanDir = fullfile(baseDir, 'clean_wav');
    if ~exist(wavCleanDir,'dir'), mkdir(wavCleanDir); end

    streams_clean = makeStreamingCorpus(cfgC, testFiles, testLabels, wavCleanDir);
    streams_clean = addStreamingWindows(streams_clean, cfgC);

    % ensure wavPath points to real files
    for s = 1:numel(streams_clean)
        if ~isfile(streams_clean(s).wavPath)
            streams_clean(s).wavPath = fullfile(wavCleanDir, sprintf('stream_%02d.wav', s));
        end
    end

    save(fullfile(baseDir, 'streams_long_clean.mat'), 'streams_clean', '-v7.3');
    fprintf('make_streams_long: wrote CLEAN -> %s\n', fullfile(baseDir,'streams_long_clean.mat'));

    % ---------------- NOISY ----------------
    cfgN = cfg;
    cfgN.streaming.numStreams     = numStreams;
    cfgN.streaming.streamLenSec   = streamLenSec;
    cfgN.streaming.winSpanMs      = spanMs;
    cfgN.streaming.hopWinMs       = hopWinMs;
    cfgN.streaming.keywordsPerMin = keywordsPerMin;
    cfgN.streaming.minGapSec      = minGapSec;
    cfgN.streaming.noiseSNRdB     = noisySNRdB;   % SNR-controlled noisy
    cfgN.streaming.bgGain         = noisyBgGain;  % fallback if SNR mixing skipped
    cfgN.sainath.labelTolMs       = labelTolMs;

    wavNoisyDir = fullfile(baseDir, 'noisy_wav');
    if ~exist(wavNoisyDir,'dir'), mkdir(wavNoisyDir); end

    streams_noisy = makeStreamingCorpus(cfgN, testFiles, testLabels, wavNoisyDir);
    streams_noisy = addStreamingWindows(streams_noisy, cfgN);

    for s = 1:numel(streams_noisy)
        if ~isfile(streams_noisy(s).wavPath)
            streams_noisy(s).wavPath = fullfile(wavNoisyDir, sprintf('stream_%02d.wav', s));
        end
    end

    save(fullfile(baseDir, 'streams_long_noisy.mat'), 'streams_noisy', '-v7.3');
    fprintf('make_streams_long: wrote NOISY -> %s\n', fullfile(baseDir,'streams_long_noisy.mat'));
end
