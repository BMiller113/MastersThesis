function net = trainCNN(XTrain, YTrain, layers, cfg)
% trainCNN — stratified holdout + recent: checkpointing/resume.

if nargin < 4 || isempty(cfg), cfg = kws_config(); end
if isfield(cfg.train,'seed'), rng(cfg.train.seed); else, rng(42); end

if ~iscategorical(YTrain), YTrain = categorical(YTrain); end
YTrain = removecats(YTrain);

if ndims(XTrain) ~= 4
    error('XTrain must be 4-D (H x W x C x N). Got %s', mat2str(size(XTrain)));
end
if ~isa(XTrain,'single'), XTrain = single(XTrain); end

% ---- Holdout ----
valFrac = getfielddef(cfg,'train','valFrac',0.20);
valIdx = stratifiedHoldoutIdx(YTrain, valFrac);

epochs      = getfielddef(cfg,'train','epochs',50);
batchSize   = getfielddef(cfg,'train','batchSize',128);
valFreqUser = getfielddef(cfg,'train','valFreq',[]);
initLR      = getfielddef(cfg,'train','initLR',1e-3);
weightDecay = getfielddef(cfg,'train','weightDecay',1e-3);

numTrain = sum(~valIdx);
% Compute iterations per epoch so validation prints once per epoch by default
itersPerEpoch = max(1, floor(numTrain / batchSize));
if isempty(valFreqUser)
    valFreq = itersPerEpoch;   % one printout per epoch
else
    valFreq = valFreqUser;     % manual override from cfg still works
end
%{
numTrain = sum(~valIdx);
if isempty(valFreqUser)
    valFreq = max(200, floor(numTrain / batchSize));
else
    valFreq = valFreqUser;
end
%}
% ---- Checkpointing ----
useCkpt = isfield(cfg.train,'enableCheckpointing') && cfg.train.enableCheckpointing;
resume  = isfield(cfg.train,'resumeFromCheckpoint') && cfg.train.resumeFromCheckpoint;
ckptDir = cfg.paths.checkpointDir;

if useCkpt && ~exist(ckptDir,'dir'), mkdir(ckptDir); end

options = trainingOptions('adam', ...    %'ExecutionEnvironment', 'gpu', ... Dali
    'ExecutionEnvironment', cfg.train.executionEnv, ... % Use the detected environment, Dali
    'MaxEpochs',            epochs, ...
    'MiniBatchSize',        batchSize, ...
    'ValidationData',       {XTrain(:,:,:,valIdx), YTrain(valIdx)}, ...
    'ValidationFrequency',  valFreq, ...
    'InitialLearnRate',     initLR, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.5, ...
    'LearnRateDropPeriod',  10, ...
    'L2Regularization',     weightDecay, ...
    'Shuffle',              'every-epoch', ...
    'Verbose',              ~cfg.runtime.suppressTrainingVerbose, ...
    'Plots',                tern(cfg.runtime.showTrainingPlots,'training-progress','none'));

if useCkpt
    options.CheckpointPath = ckptDir;
    if isfield(cfg.train,'checkpointFrequency') && ~isempty(cfg.train.checkpointFrequency)
        options.CheckpointFrequency = cfg.train.checkpointFrequency;
    end
end

Xtr = XTrain(:,:,:,~valIdx);
Ytr = YTrain(~valIdx);

% ---- Resume ----
initialNet = [];
if useCkpt && resume
    initialNet = tryLoadLatestCheckpoint(ckptDir);
end

if ~isempty(initialNet)
    fprintf('Resuming from latest checkpoint in %s\n', ckptDir);
    try
        net = trainNetwork(Xtr, Ytr, layers, options, 'InitialNetwork', initialNet);
    catch
        % older MATLAB: no InitialNetwork support
        warning('Resume not supported on this MATLAB version; training from scratch.');
        net = trainNetwork(Xtr, Ytr, layers, options);
    end
else
    net = trainNetwork(Xtr, Ytr, layers, options);
end

% Keep only a few newest checkpoints
if useCkpt && isfield(cfg.train,'maxCheckpointsToKeep') && cfg.train.maxCheckpointsToKeep > 0
    pruneCheckpoints(ckptDir, cfg.train.maxCheckpointsToKeep);
end

end

function tf = tern(cond,a,b), if cond, tf=a; else, tf=b; end, end

function valIdx = stratifiedHoldoutIdx(Y, frac)
Y = removecats(Y);
cats = categories(Y);
n = numel(Y);
valIdx = false(n,1);
for c = 1:numel(cats)
    idx = find(Y == cats{c});
    nc  = numel(idx);
    if nc == 1
        nVal = 0;
    else
        nVal = max(1, round(frac*nc));
        nVal = min(nVal, nc-1);
    end
    if nVal > 0
        sel = idx(randperm(nc, nVal));
        valIdx(sel) = true;
    end
end
end

function v = getfielddef(cfg, section, name, default)
v = default;
if isstruct(cfg) && isfield(cfg, section)
    S = cfg.(section);
    if isfield(S, name) && ~isempty(S.(name)), v = S.(name); end
end
end

function net = tryLoadLatestCheckpoint(ckptDir)
net = [];
d = dir(fullfile(ckptDir, 'net_checkpoint__*.mat'));
if isempty(d), return; end
[~,ix] = max([d.datenum]);
S = load(fullfile(d(ix).folder, d(ix).name));
% MATLAB checkpoint files typically store "net"
if isfield(S,'net'), net = S.net; end
end

function pruneCheckpoints(ckptDir, keepN)
d = dir(fullfile(ckptDir, 'net_checkpoint__*.mat'));
if numel(d) <= keepN, return; end
[~,ord] = sort([d.datenum],'descend');
kill = d(ord(keepN+1:end));
for i = 1:numel(kill)
    try, delete(fullfile(kill(i).folder, kill(i).name)); catch, end
end
end
