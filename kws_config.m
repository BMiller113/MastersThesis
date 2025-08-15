function cfg = kws_config(profile)
% kws_config: Central config for the Keyword Search project 8/15
% Usage:
%   cfg = kws_config();           % defaults
%   cfg = kws_config('fast');     % example alt profile

if nargin < 1, profile = 'default'; end

% -------- Paths --------
cfg.paths.datasetRoot   = 'C:\Users\bjren\MATLAB\Projects\KeywordSpottingThesis\Data\Kaggle_ GoogleSpeechCommandsV2';
cfg.paths.genderMapFile = 'speakerGenderMap.mat';
cfg.paths.outputDir     = pwd;   % where results_*.mat, csvs are saved

% -------- Runtime/UI --------
cfg.runtime.makePlots        = false;   % ROC/figures during runs
cfg.runtime.figureVisibility = 'off';   % 'off' during long runs
cfg.runtime.suppressWarnings = false;   % true for quiet runs

% -------- Experiment targets --------
cfg.experiments.includeModes = {'none','filter','filter+mel'};     % which high-level modes to run
cfg.experiments.gendersToRun = {'all','male','female'};            % which groups to include
cfg.experiments.melModes     = {'default','narrow','wide','prop7k','prop8k'}; % mel variants, prop7 and prop8 being the proportional increases suggested by Dr. Wang in Late July

% Force a target class or threshold (leave [] for data-driven)
cfg.experiments.forcePosLabel = [];    % example: 'yes'
cfg.experiments.fixedThreshold = [];   % example: 0.7

% -------- Features (extractor currently owns these) --------
cfg.features.baseBands     = 40;   % baseline mel bands (prop modes scale this)
cfg.features.targetFrames  = 32;   % time steps kept constant
cfg.features.frameMs       = 25;
cfg.features.hopMs         = 10;

% -------- Model --------
cfg.model.arch = 'tpool2';          % 'tpool2' | 'one-fstride4' | 'trad-fpool3'

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


%To build gender map run the below:
%
% datasetRoot = kws_config().paths.datasetRoot;
% genderMap = build_speaker_gender_map(datasetRoot);
% save('speakerGenderMap.mat','genderMap');   % path must match cfg.paths.genderMapFile
%
% To begin getting results run main.m after adjusting all configs to suit
% your setup
