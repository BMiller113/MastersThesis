function net = trainCNN(XTrain, YTrain, layers, cfg)
% trainCNN — GPU-enabled training with stratified holdout, sane defaults,
% and low-overhead options for faster single-GPU runs.

    % ----------------- Setup & RNG -----------------
    if nargin < 4 || isempty(cfg), cfg = kws_config(); end
    if isfield(cfg.train,'seed'), rng(cfg.train.seed); else, rng(42); end

    % ----------------- Labels -----------------
    if ~iscategorical(YTrain), YTrain = categorical(YTrain); end
    YTrain = removecats(YTrain);

    if numel(categories(YTrain)) < 2
        error('trainCNN:needTwoClasses', ...
            'Training requires at least two label categories. Got: %s', strjoin(string(categories(YTrain))));
    end

    % ----------------- Data sanity -----------------
    % Make sure data is 4-D: H x W x C x N and single precision.
    if ndims(XTrain) ~= 4
        error('trainCNN:badDims', 'XTrain must be 4-D (H x W x C x N). Got size: %s', mat2str(size(XTrain)));
    end
    if ~isa(XTrain, 'single')
        XTrain = single(XTrain);
    end

    % If your first layer has a fixed input size, validate it here.
    try
        inSz = layers(1).InputSize; % e.g., [40 32 1]
        xSz  = size(XTrain);
        if numel(inSz) >= 3
            need = inSz(1:3);
            have = xSz(1:3);
            if ~isequal(need, have)
                error(['Input size mismatch.\n' ...
                       '  Network expects: %s\n  XTrain has:     %s\n' ...
                       'Tip: ensure your extractor makes %dx%dx%d (e.g., 40x32x1) stacks.'], ...
                       mat2str(need), mat2str(have), need(1), need(2), need(3));
            end
        end
    catch
        % If layers(1).InputSize isn’t available, skip the check.
    end

    % ----------------- Stratified Holdout -----------------
    if isfield(cfg.train,'valFrac'), valFrac = cfg.train.valFrac; else, valFrac = 0.20; end
    valIdx  = stratifiedHoldoutIdx(YTrain, valFrac);

    % Safety fallback if stratification degenerates
    if all(~valIdx) || all(valIdx) || numel(unique(YTrain(~valIdx))) < 2
        warning('trainCNN:holdoutFallback', ...
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

    % ----------------- Hyperparameters -----------------
    epochs      = getfielddef(cfg,'train','epochs',50);

    % ↑ Bump default batch size; still respects cfg.train.batchSize if set
    batchSize   = getfielddef(cfg,'train','batchSize',384);   % try 384; go 512 if it fits

    % Temporary placeholder; we'll recompute valFreq after we know numTrain & batchSize
    valFreqUser = getfielddef(cfg,'train','valFreq',[]);
    initLR      = getfielddef(cfg,'train','initLR',1e-3);
    weightDecay = getfielddef(cfg,'train','weightDecay',1e-3);

    % ----------------- Derive a better ValidationFrequency -----------------
    % If user didn't explicitly set valFreq, validate ~once per epoch.
    numTrain = sum(~valIdx);
    if isempty(valFreqUser)
        valFreq = max(200, floor(numTrain / batchSize));
    else
        valFreq = valFreqUser;
    end

    % ----------------- trainingOptions (GPU) -----------------
    % If you get OOM, reduce MiniBatchSize (e.g., 256, 128, or 64).
    options = trainingOptions('adam', ...
        'ExecutionEnvironment', 'gpu', ...
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
        'Verbose',              true, ...
        'VerboseFrequency',     200, ...   % reduce console I/O
        'Plots',                'none');   % low-overhead (set to 'training-progress' if you prefer)

    % ----------------- Train -----------------
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
