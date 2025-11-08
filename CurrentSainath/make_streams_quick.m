function make_streams_quick()
% make_streams_quick
% Build two small streaming sets:
%   - clean : 6 streams x 90 s
%   - noisy : 6 streams x 90 s, SNR=10 dB
% Sainath geometry: 25ms/10ms, 23L+1+8R -> ~335 ms span, 10 ms decisions
%
% Saves to:
%   <pwd>\streams_quick\streams_quick_clean.mat
%   <pwd>\streams_quick\streams_quick_noisy.mat

    cfg = kws_config('sainath14');

    numStreams   = 6;
    streamLenSec = 90;
    frameMs      = cfg.features.frameMs;   % 25
    hopMs        = cfg.features.hopMs;     % 10
    spanMs       = frameMs + hopMs*(23+8); % 335
    hopWinMs     = 10;                     % decisions every 10 ms

    [~, ~, testFiles, testLabels] = loadAudioData();

    baseDir = fullfile(pwd, 'streams_quick');
    if ~exist(baseDir,'dir'), mkdir(baseDir); end

    % ---------------- CLEAN ----------------
    cfgC = cfg;
    cfgC.streaming.numStreams   = numStreams;
    cfgC.streaming.streamLenSec = streamLenSec;
    cfgC.streaming.winSpanMs    = spanMs;
    cfgC.streaming.hopWinMs     = hopWinMs;
    cfgC.streaming.noiseSNRdB   = [];
    cfgC.streaming.bgGain       = 0.0;

    wavCleanDir = fullfile(baseDir, 'clean_wav');
    if ~exist(wavCleanDir,'dir'), mkdir(wavCleanDir); end

    streams_clean = makeStreamingCorpus(cfgC, testFiles, testLabels, wavCleanDir);
    streams_clean = addStreamingWindows(streams_clean, cfgC);

    % make sure wavPath is actual file
    for s = 1:numel(streams_clean)
        if ~isfile(streams_clean(s).wavPath)
            streams_clean(s).wavPath = fullfile(wavCleanDir, sprintf('stream_%02d.wav', s));
        end
    end

    save(fullfile(baseDir, 'streams_quick_clean.mat'), 'streams_clean', '-v7.3');
    fprintf('make_streams_quick: wrote CLEAN -> %s\n', fullfile(baseDir, 'streams_quick_clean.mat'));

    % ---------------- NOISY ----------------
    cfgN = cfg;
    cfgN.streaming.numStreams   = numStreams;
    cfgN.streaming.streamLenSec = streamLenSec;
    cfgN.streaming.winSpanMs    = spanMs;
    cfgN.streaming.hopWinMs     = hopWinMs;
    cfgN.streaming.noiseSNRdB   = 10;   % light noise
    cfgN.streaming.bgGain       = 0.3;

    wavNoisyDir = fullfile(baseDir, 'noisy_wav');
    if ~exist(wavNoisyDir,'dir'), mkdir(wavNoisyDir); end

    streams_noisy = makeStreamingCorpus(cfgN, testFiles, testLabels, wavNoisyDir);
    streams_noisy = addStreamingWindows(streams_noisy, cfgN);

    for s = 1:numel(streams_noisy)
        if ~isfile(streams_noisy(s).wavPath)
            streams_noisy(s).wavPath = fullfile(wavNoisyDir, sprintf('stream_%02d.wav', s));
        end
    end

    save(fullfile(baseDir, 'streams_quick_noisy.mat'), 'streams_noisy', '-v7.3');
    fprintf('make_streams_quick: wrote NOISY -> %s\n', fullfile(baseDir, 'streams_quick_noisy.mat'));
end
