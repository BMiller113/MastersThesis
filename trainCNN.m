function net = trainCNN(XTrain, YTrain, layers, cfg)
    % Stratified Holdout

    if nargin < 4 || isempty(cfg), cfg = kws_config(); end
    if isfield(cfg.train,'seed'), rng(cfg.train.seed); else, rng(42); end

    % Clean categorical labels
    if ~iscategorical(YTrain), YTrain = categorical(YTrain); end
    YTrain = removecats(YTrain);

    % Build a stratified holdout > trying to not drop class from training
    if isfield(cfg.train,'valFrac'), valFrac = cfg.train.valFrac; else, valFrac = 0.20; end
    valIdx  = stratifiedHoldoutIdx(YTrain, valFrac);

    % Safety
    if all(~valIdx) || all(valIdx) || numel(unique(YTrain(~valIdx))) < 2
        warning('trainCNN:holdoutFallback',...
            'Stratified split degenerated; falling back to random non-stratified holdout.');
        n = numel(YTrain);
        valIdx = false(n,1);
        valIdx(randperm(n, max(1, round(valFrac*n)))) = true;
        % ensure at least two classes in training
        if numel(unique(YTrain(~valIdx))) < 2
            C = categories(YTrain);
            for k = 1:numel(C)
                if ~any(YTrain(~valIdx)==C{k}) && any(YTrain(valIdx)==C{k})
                    j = find(YTrain==C{k} & valIdx, 1, 'first');
                    valIdx(j) = false;
                end
            end
        end
    end

    % Training options
    epochs      = getfielddef(cfg,'train','epochs',50);
    batchSize   = getfielddef(cfg,'train','batchSize',128);
    valFreq     = getfielddef(cfg,'train','valFreq',30);
    initLR      = getfielddef(cfg,'train','initLR',1e-3);
    weightDecay = getfielddef(cfg,'train','weightDecay',1e-3);

    options = trainingOptions('adam', ...
        'MaxEpochs',           epochs, ...
        'MiniBatchSize',       batchSize, ...
        'ValidationData',      {XTrain(:,:,:,valIdx), YTrain(valIdx)}, ...
        'ValidationFrequency', valFreq, ...
        'InitialLearnRate',    initLR, ...
        'LearnRateSchedule',   'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'L2Regularization',    weightDecay, ...
        'Shuffle',             'every-epoch', ...
        'Verbose',             false, ...
        'Plots',               'none');

    net = trainNetwork(XTrain(:,:,:,~valIdx), YTrain(~valIdx), layers, options);
end

% Keeps ≥1 training sample per class
function valIdx = stratifiedHoldoutIdx(Y, frac)
    Y = removecats(Y);
    cats = categories(Y);
    n = numel(Y);
    valIdx = false(n,1);
    for c = 1:numel(cats)
        idx = find(Y == cats{c});
        nc  = numel(idx);
        if nc == 1
            nVal = 0;                    % keep the only sample for training
        else
            nVal = max(1, round(frac*nc));
            nVal = min(nVal, nc-1);      % leave at least 1 for training
        end
        if nVal > 0
            sel = idx(randperm(nc, nVal));
            valIdx(sel) = true;
        end
    end
end

% Helper for reading config files
function v = getfielddef(cfg, section, name, default)
    v = default;
    if isstruct(cfg) && isfield(cfg, section)
        S = cfg.(section);
        if isfield(S, name) && ~isempty(S.(name)), v = S.(name); end
    end
end
