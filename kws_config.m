function cfg = kws_config(profile)
% kws_config: Central config for the Keyword Search project 8/15
% Usage:
%   cfg = kws_config();           % defaults
%   cfg = kws_config('fast');     % example alt profile

if nargin < 1, profile = 'default'; end

% -------- Paths --------
cfg.paths.datasetRoot   = 'C:\Users\bjren\MATLAB\Projects\KeywordSpottingThesis\Data\Kaggle_ GoogleSpeechCommandsV2';
cfg.paths.genderMapFile = 'speakerGenderMap.mat';
cfg.paths.outputDir     = 'Results';   % where results_*.mat, csvs are saved, was 'pwd' 9/8

% -------- Runtime/UI --------
cfg.runtime.makePlots         = false;   % ROC/figures during runs
cfg.runtime.figureVisibility  = 'off';   % 'off' during long runs
cfg.runtime.suppressWarnings  = false;   % true for quiet runs
cfg.runtime.showTrainingPlots = false;   % training progress window popup ('training-progress') (NEW: moved here)

% -------- Experiment targets --------
cfg.experiments.includeModes = {'none', 'mel-only'};     % {'none','filter','filter+mel','mel-only'} which high-level modes to run
cfg.experiments.gendersToRun = {'all','male','female'};            % which groups to include   {'all','male','female'};  
cfg.experiments.melModes     = {'default','narrow','wide','prop7k','prop8k'}; % {'default','narrow','wide','prop7k','prop8k'}
                                                                              % mel variants, prop7 and prop8 being the proportional increases suggested by Dr. Wang in Late July

cfg.experiments.enableLinearForFemale = false; % true, false. Linear for female as proposed by Dr. Wang Late July

% Force a target class or threshold (leave [] for data-driven)
cfg.experiments.forcePosLabel = [];    % example: 'yes'
cfg.experiments.fixedThreshold = [];   % example: 0.7

% -------- Features (extractor currently owns these) --------
cfg.features.baseBands     = 80;   % baseline mel bands (prop modes scale this) %default 40
cfg.features.targetFrames  = 98;   % time steps kept constant (SET TO 64/98 TO COVER ~1s), was 32 9/8 was 98 9/17 101 9/19
cfg.features.frameMs       = 25;   % analysis window size in ms
cfg.features.hopMs         = 10;   % desired hop (ms) between frames; actual overlap = frameMs - hopMs (NEW)
% Optional: cropping policy if utterance longer than targetFrames (NEW)
cfg.features.timeCrop      = 'center';  % 'center' | 'left' | 'right'

% -------- Warden pattern --------
cfg.warden.enable          = true;      % turn on/off the Warden loader path
cfg.warden.targetWords     = {'yes','no','up','down','left','right','on','off','stop','go'};
cfg.warden.unknownPct      = 10;        % % of the *final* set to be _unknown_
cfg.warden.silencePct      = 10;        % % of the *final* set to be _silence_
cfg.warden.timeShiftMs     = 100;       % +/- time shift (train only)
cfg.warden.bgFreq          = 0.8;       % prob to mix background noise (train only)
cfg.warden.bgVolRange      = [0.0 0.1]; % mix gain (train only)

% Force Warden Setuo
cfg.features.baseBands     = 40;  % Warden/TF reference uses 40 MFCC / 40 log-mel
cfg.features.frameMs       = 30;  % window = 30 ms
cfg.features.hopMs         = 10;  % hop    = 10 ms
cfg.features.targetFrames  = 98;  % ~ 1 + floor((1000-30)/10) = 98 frames (≈ 1s)

% If any files aren’t 16 kHz force resample during extraction:
cfg.features.forceSampleRate = 16000;   % optional; extractor can honor this if you add it

% -------- Model --------
cfg.model.arch = 'trad-fpool3';          % 'tpool2' | 'one-fstride4' | 'trad-fpool3'

% -------- Training --------
cfg.train.epochs       = 50;
cfg.train.batchSize    = 128;
cfg.train.valFrac      = 0.20;      % used by custom stratified splitter
cfg.train.initLR       = 1e-3;
cfg.train.weightDecay  = 1e-3;
cfg.train.seed         = 42;
cfg.train.valFreq      = 30;

% -------- Evaluation --------
cfg.eval.perClassCSV   = true;      % write results_by_class_*.csv if helper exists

% -------- DET / ROC plotting --------
cfg.plots.roc.xAxis = 'fpr_percent';   % recommended for utterance ROC
cfg.plots.roc.xlim  = [0 5];           % zoom x to 0–5% FPR
cfg.plots.roc.ylim  = [0 20];          % zoom y to 0–20% FRR
cfg.plots.roc.addDET = true;           % also show DET
% cfg.plots.frameHopSec = 0.01;        % only used if xAxis='fa_per_hour'

% (Optional) Include a separate baseline folder to overlay on plots (NEW)
cfg.plots.roc.includeBaseline     = false;
cfg.plots.roc.includeBaselineFrom = fullfile(pwd,'BaselineResults');

% -------- Profiles --------
switch lower(profile)
    case 'fast'   % quicker smoke test
        cfg.runtime.makePlots  = false;
        cfg.experiments.includeModes = {'none','filter+mel'};
        cfg.experiments.gendersToRun = {'male','female'};
        cfg.experiments.melModes     = {'default','prop7k'};
        cfg.train.epochs    = 10;
        cfg.train.batchSize = 256;
    otherwise
        % default stays as set above
end
end
