function cfg = kws_config(profile)
% kws_config: Central config for Keyword Spotting thesis experiments

% profile = 'default' is changed to 'fast' just for functional check; change it back once code runs correctly 
%if nargin < 1, profile = 'fast'; end
if nargin < 1, profile = 'default'; end

% -------- Project root --------
thisFile = mfilename('fullpath');
cfg.paths.projectRoot = fileparts(thisFile);

% -------- Dataset roots --------
cfg.paths.datasetRootV2   = 'C:\Users\bjren\MATLAB\Projects\KeywordSpottingThesis\Data\Kaggle_GoogleSpeechCommandsV2';
%cfg.paths.datasetRootV2   = 'C:\temp\Kaggle_GoogleSpeechCommandsV2'; %Dali

cfg.paths.datasetRootV1   = 'C:\Users\bjren\MATLAB\Projects\KeywordSpottingThesis\Data\Kaggle_GoogleSpeechCommandsV1';
%cfg.paths.datasetRootV1   = 'C:\Temp\Kaggle_GoogleSpeechCommandsV1'; %Dali


cfg.dataset.version       = 'v2';      % 'v1' or 'v2'
cfg.dataset.combineV1V2   = false;

% -------- Output paths --------

    %cfg.paths.outputDir = 'C:\temp\Results'; %Dali
    cfg.paths.outputDir       = fullfile(cfg.paths.projectRoot, 'Results');

    % The rest of your paths will automatically follow this root
    cfg.paths.cacheDir        = fullfile(cfg.paths.outputDir, 'cache'); %Dali
    cfg.paths.modelDir        = fullfile(cfg.paths.outputDir, 'models'); %Dali

%cfg.paths.outputDir       = fullfile(cfg.paths.projectRoot, 'Results');
%cfg.paths.cacheDir        = fullfile(cfg.paths.outputDir, 'cache'); Dali
%cfg.paths.modelDir        = fullfile(cfg.paths.outputDir, 'models'); Dali
cfg.paths.metricsDir      = fullfile(cfg.paths.outputDir, 'metrics');
cfg.paths.checkpointDir   = fullfile(cfg.paths.outputDir, 'checkpoints');

% -------- Hardware Detection -------- added Dali
try
    if gpuDeviceCount > 0
        cfg.train.executionEnv = 'gpu'; % Use GPU if available
    else
        cfg.train.executionEnv = 'cpu'; % Fallback to CPU
    end
catch
    cfg.train.executionEnv = 'cpu';
end
% -------- Runtime/UI --------
cfg.runtime.makePlots                = false;
cfg.runtime.figureVisibility         = 'off';
cfg.runtime.suppressWarnings         = false;
cfg.runtime.showTrainingPlots        = false;
cfg.runtime.quiet                    = true;
cfg.runtime.suppressTrainingVerbose  = true;

% -------- Experiment targets --------
cfg.experiments.includeModes = {'none', 'mel-only'};    % pooled-only
cfg.experiments.gendersToRun = {'all'};                 % pooled-only
cfg.experiments.melModes     = {'default','narrow','wide','prop7k','prop8k'};
cfg.experiments.enableLinearForFemale = false; % legacy, probably don't touch

% Force a target class or threshold (leave [] for data-driven)
cfg.experiments.forcePosLabel  = [];
cfg.experiments.fixedThreshold = [];

% -------- Features --------
cfg.features.baseBands       = 40; %40 or 80
cfg.features.targetFrames    = 98; %32 or 98
cfg.features.frameMs         = 30;
cfg.features.hopMs           = 10;
cfg.features.timeCrop        = 'center';
cfg.features.forceSampleRate = 16000;

% -------- Model --------
cfg.model.arch = 'one-fstride4';   % 'tpool2' | 'one-fstride4' | 'trad-fpool3'

% -------- Training --------
cfg.train.epochs       = 50;
cfg.train.batchSize    = 128;
cfg.train.valFrac      = 0.20;
cfg.train.initLR       = 1e-3;
cfg.train.weightDecay  = 1e-3;
cfg.train.seed         = 42;
cfg.train.valFreq      = []; % change from 30; [] = auto (one printout per epoch)

cfg.train.enableCheckpointing   = true;
cfg.train.checkpointFrequency   = 200;
cfg.train.resumeFromCheckpoint  = true;
cfg.train.maxCheckpointsToKeep  = 3;

% -------- Caching --------
cfg.cache.enableFeatureCache = true;
cfg.cache.enableModelCache   = true;

% -------- Evaluation / exports --------
cfg.eval.exportCSV       = true;
cfg.eval.exportMAT       = true;
cfg.eval.perClassCSV     = true;
cfg.eval.confusionMatrix = true;
cfg.eval.topK            = [1 3 5];
cfg.eval.computePRF1     = true;
cfg.eval.computeOvRAUC   = true;

% -------- ROC plotting / saving --------
cfg.plots.roc.popUpAtEnd    = true;
cfg.plots.roc.saveToDisk    = true;
cfg.plots.roc.outSubdir     = 'roc';
cfg.plots.roc.style         = 'roc';
cfg.plots.roc.xlimPercent   = [0 5];

% -------- Overlay ROC at end --------
cfg.plots.overlay.enable        = true;
cfg.plots.overlay.popUpAtEnd    = true;
cfg.plots.overlay.saveToDisk    = true;
cfg.plots.overlay.outSubdir     = cfg.plots.roc.outSubdir;  % same folder as ROC
cfg.plots.overlay.style         = 'roc';
cfg.plots.overlay.xlimPercent   = [0 5];

% -------- Profiles --------
switch lower(profile)
    case 'fast'
        cfg.features.baseBands       = 40; % added for fast execution; Dali
        cfg.experiments.includeModes = {'none'}; % from: {'none','mel-only'};
        cfg.experiments.gendersToRun = {'all'};
        cfg.experiments.melModes     = {'default'}; % changed from {'default','prop7k'}; Dali
        cfg.train.epochs    = 25;% added to reduce running time
        cfg.train.batchSize = 256;
        cfg.runtime.suppressTrainingVerbose = false;  % added to track progress; Dali
    case 'long1s'
        cfg.features.baseBands     = 40;
        cfg.features.frameMs       = 25;
        cfg.features.hopMs         = 10;
        cfg.features.targetFrames  = 98;

    otherwise
        % keep defaults
end

% Ensure output folders exist
ensureDir(cfg.paths.outputDir);
ensureDir(cfg.paths.cacheDir);
ensureDir(cfg.paths.modelDir);
ensureDir(cfg.paths.metricsDir);
ensureDir(cfg.paths.checkpointDir);

end

function ensureDir(p)
if ~exist(p,'dir'), mkdir(p); end
end
