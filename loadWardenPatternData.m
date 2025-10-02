function [XTrain, YTrain, XTest, YTest] = loadWardenPatternData(cfg)
% loadWardenPatternData: Recreate the Warden/TF Speech Commands data pattern.
%  - 12 classes: 10 keywords + _unknown_ + _silence_
%  - Uses testing_list.txt / validation_list.txt
%  - Maps all non-target words to _unknown_
%  - Generates synthetic _silence_ from _background_noise_ wavs
%  - Optional train-time augmentation: time shift + background mix (SNR or fixed gain)
%  - Outputs come already feature-extracted via your extractFeatures.

    % Guards, defaults
    assert(isfield(cfg,'paths') && isfield(cfg.paths,'datasetRoot'), ...
        'cfg.paths.datasetRoot must be set.');
    root = cfg.paths.datasetRoot;

    % Targets (Warden-10 if not provided)
    if ~isfield(cfg,'warden') || ~isfield(cfg.warden,'targetWords') || isempty(cfg.warden.targetWords)
        cfg.warden.targetWords = {'yes','no','up','down','left','right','on','off','stop','go'};
    end
    targets = string(cfg.warden.targetWords(:)');
    targets = strtrim(targets);
    targets = targets(targets ~= "");
    assert(~isempty(targets), 'cfg.warden.targetWords had no valid entries.');

    % Percentages & augmentation defaults
    if ~isfield(cfg.warden,'silencePct'),  cfg.warden.silencePct = 10; end    % % of final set
    if ~isfield(cfg.warden,'unknownPct'),  cfg.warden.unknownPct = 10; end    % % of final set
    if ~isfield(cfg.warden,'timeShiftMs'), cfg.warden.timeShiftMs = 100; end
    if ~isfield(cfg.warden,'bgFreq'),      cfg.warden.bgFreq      = 0.5;  end % probability of bg mix
    if ~isfield(cfg.warden,'bgVolRange'),  cfg.warden.bgVolRange  = [0 0.1]; end
    if ~isfield(cfg.warden,'useSnrMix'),   cfg.warden.useSnrMix   = false; end
    if ~isfield(cfg.warden,'snrDbRange'),  cfg.warden.snrDbRange  = [5 20];   end

    %Read official split lists
    [trainFilesAll, trainLabelsAll, testFilesAll, testLabelsAll] = loadAudioData(); %#ok<ASGLU>

    %Map to 12-class space (targets + _unknown_)
    [trainFiles, trainLabels] = mapToWardenLabels(trainFilesAll, root, targets);
    [testFiles,  testLabels ] = mapToWardenLabels(testFilesAll,  root, targets);

    % Create silence wavs from _background_noise_ and append
    silPct = clamp01(cfg.warden.silencePct/100);
    unkPct = clamp01(cfg.warden.unknownPct/100);

    tmpSil = fullfile(pwd,'tmp_silence');
    if ~exist(tmpSil,'dir'), mkdir(tmpSil); end

    % We want final SILENCE% = silPct of total:
    % Let N be current count; S_final = silPct/(1 - silPct) * N
    numSilTrain = (silPct>0) * round(silPct/(1-silPct) * numel(trainFiles));
    numSilTest  = (silPct>0) * round(silPct/(1-silPct) * numel(testFiles));

    silTrainFiles = synthSilenceWavs(root, tmpSil, numSilTrain, 1000);  % 1s clips
    silTestFiles  = synthSilenceWavs(root, tmpSil, numSilTest,  1000);

    trainFiles = [trainFiles; silTrainFiles];
    trainLabels = [trainLabels; repmat(categorical("_silence_"), numel(silTrainFiles),1)];
    testFiles  = [testFiles;  silTestFiles ];
    testLabels = [testLabels; repmat(categorical("_silence_"),  numel(silTestFiles),1)];

    % UNKNOWN percentage enforcement by down/up-sampling
    [trainFiles, trainLabels] = enforceUnknownPct(trainFiles, trainLabels, unkPct);
    [testFiles,  testLabels ] = enforceUnknownPct(testFiles,  testLabels,  unkPct);

    % Train-time augmentation/ time shift + background mix
    if needAugment(cfg)
        augTmp = fullfile(pwd,'tmp_aug');
        if ~exist(augTmp,'dir'), mkdir(augTmp); end
        trainFilesAug = augmentToTempWavs(trainFiles, root, augTmp, cfg);
    else
        trainFilesAug = trainFiles;
    end

    % Feature extraction
    [XTrain, validTrain] = extractFeatures(trainFilesAug, 'all', 'default', cfg);
    [XTest,  validTest ] = extractFeatures(testFiles,     'all', 'default', cfg);

    YTrain = trainLabels(validTrain);
    YTest  = testLabels(validTest);

    % Exactly the 12-class label space & order ----
    catNames = [cellstr(lower(string(targets))), {'_unknown_','_silence_'}];
    YTrain = addcats(YTrain, setdiff(catNames, categories(YTrain)));
    YTest  = addcats(YTest,  setdiff(catNames, categories(YTest)));
    YTrain = reordercats(YTrain, catNames);
    YTest  = reordercats(YTest,  catNames);
end

% ===================== Helpers =====================

function [filesOut, labelsOut] = mapToWardenLabels(filesIn, root, targets)
    n = numel(filesIn);
    filesOut  = cell(n,1);
    labelsOut = strings(n,1);
    targetSet = lower(string(targets));
    for i = 1:n
        f = filesIn{i};
        rel = strrep(f, [root filesep], '');
        rel = strrep(rel, '\','/');  % normalize
        parts = split(rel, '/');
        lab = lower(string(parts{1}));  % folder name = word
        if any(strcmp(lab, targetSet))
            labelsOut(i) = lab;
        else
            labelsOut(i) = "_unknown_";
        end
        filesOut{i} = f;
    end
    labelsOut = categorical(labelsOut);
end

function filesOut = synthSilenceWavs(root, outDir, count, clipMs)
    filesOut = cell(count,1);
    if count==0, return; end

    noiseDir = fullfile(root,'_background_noise_');
    noiseList = dir(fullfile(noiseDir,'*.wav'));
    sr = 16000; L = round(sr*clipMs/1000);

    if isempty(noiseList)
        % Fallback: zeros if background noise missing
        for i = 1:count
            seg = zeros(L,1,'double');
            mx = max(1.0, max(abs(seg)));
            seg = 0.99 * (seg./mx);
            p = fullfile(outDir, sprintf('sil_%06d.wav', i));
            audiowrite(p, seg, sr);
            filesOut{i} = p;
        end
        return;
    end

    rng(123);  % reproducible
    for i = 1:count
        k = randi(numel(noiseList));
        [y, fs] = audioread(fullfile(noiseList(k).folder, noiseList(k).name));
        if size(y,2) > 1, y = mean(y,2); end
        if fs ~= sr, y = resample(y, sr, fs); end
        if numel(y) < L, y = repmat(y, ceil(L/numel(y)), 1); end
        start = randi(max(1, numel(y)-L+1));
        seg = y(start:start+L-1);

        % Peak-normalize to avoid clipping warnings (this was big issue
        % with matlabs given code)
        mx = max(1.0, max(abs(seg)));
        seg = 0.99 * (seg./mx);

        p = fullfile(outDir, sprintf('sil_%06d.wav', i));
        audiowrite(p, seg, sr);
        filesOut{i} = p;
    end
end

function [filesB, labelsB] = enforceUnknownPct(filesA, labelsA, unkPct)
    % Make _unknown_ approximately unkPct of the final set.
    filesB = filesA; labelsB = labelsA;
    labs = string(labelsA);
    unkMask = labs == "_unknown_";
    N = numel(labelsA);
    U = sum(unkMask);
    targetU = round(unkPct * N);

    if U > targetU
        idxAll = (1:N)'; idxUnknown = idxAll(unkMask);
        keepUnknown = idxUnknown(randperm(numel(idxUnknown), targetU));
        idxKeep = sort([idxAll(~unkMask); keepUnknown]);
        filesB  = filesA(idxKeep);
        labelsB = labelsA(idxKeep);
    elseif U < targetU && U>0
        addK = targetU - U;
        idxUnknown = find(unkMask);
        dup = idxUnknown(randi(numel(idxUnknown), addK, 1));
        filesB  = [filesA;  filesA(dup)];
        labelsB = [labelsA; labelsA(dup)];
    end
end

function tf = needAugment(cfg)
    tf = isfield(cfg,'warden') && (cfg.warden.bgFreq>0 || cfg.warden.timeShiftMs>0);
end

function filesOut = augmentToTempWavs(filesIn, root, outDir, cfg)
    % Train-only augmentation: time shift (+/- timeShiftMs) and optional background mix
    sr = 16000;            % target SR
    L  = sr * 1.0;         % always 1s clips for Warden
    N  = numel(filesIn);
    filesOut = cell(size(filesIn));

    % Precompute background list once
    noiseDir   = fullfile(root,'_background_noise_');
    noiseList  = dir(fullfile(noiseDir,'*.wav'));
    noisePaths = arrayfun(@(d) fullfile(d.folder,d.name), noiseList, 'UniformOutput', false);

    % Pull augmentation knobs locally
    timeShiftMs = cfg.warden.timeShiftMs;
    bgFreq      = cfg.warden.bgFreq;
    useSnrMix   = cfg.warden.useSnrMix;
    snrDbRange  = cfg.warden.snrDbRange;
    bgVolRange  = cfg.warden.bgVolRange;

    % Use parfor
    usePar = ~isempty(ver('parallel')) && license('test','Distrib_Computing_Toolbox');
    if usePar
        parfor i = 1:N
            filesOut{i} = augmentOneFile(filesIn{i}, outDir, sr, L, ...
                                         timeShiftMs, bgFreq, useSnrMix, snrDbRange, bgVolRange, ...
                                         noisePaths);
        end
    else
        for i = 1:N
            filesOut{i} = augmentOneFile(filesIn{i}, outDir, sr, L, ...
                                         timeShiftMs, bgFreq, useSnrMix, snrDbRange, bgVolRange, ...
                                         noisePaths);
        end
    end
end

function y = clamp01(x)
    y = max(0, min(1, x));
end

% ---------- subfunction used by parfor (NOT nested, nested no good) ----------
function outPath = augmentOneFile(inPath, outDir, sr, L, timeShiftMs, bgFreq, useSnrMix, snrDbRange, bgVolRange, noisePaths)
    [x, fs] = audioread(inPath);
    if size(x,2) > 1, x = mean(x,2); end
    if fs ~= sr, x = resample(x, sr, fs); end
    if numel(x) < L, x = [x; zeros(L-numel(x),1)]; end
    if numel(x) > L, x = x(1:L); end

    % Time shift (circular)
    shift = round( (rand()*2-1) * sr * timeShiftMs/1000 );
    if shift ~= 0, x = circshift(x, shift); end

    % Background mix
    if ~isempty(noisePaths) && rand() < bgFreq
        k = randi(numel(noisePaths));
        [n, nfs] = audioread(noisePaths{k});
        if size(n,2) > 1, n = mean(n,2); end
        if nfs ~= sr, n = resample(n, sr, nfs); end
        if numel(n) < L, n = repmat(n, ceil(L/numel(n)),1); end
        start = randi(max(1,numel(n)-L+1));
        n = n(start:start+L-1);

        if useSnrMix
            snrDb = mean(snrDbRange);
            alpha = (rms(x)/max(rms(n),eps)) * 10^(-snrDb/20);
            x = x + alpha*n;
        else
            g = bgVolRange(1) + diff(bgVolRange)*rand();
            x = x + g*n;
        end
    end

    % Peak protect ("Data clipped when writing file" warnings)
    mx = max(1.0, max(abs(x)));
    x = 0.99 * (x./mx);

    outPath = fullfile(outDir, sprintf('aug_%s.wav', char(java.util.UUID.randomUUID)));
    audiowrite(outPath, x, sr);
end
