function [X, Y, cachePath] = getOrBuildFeatures(modeTag, fileList, labels, cfg)
% Build 40x32x1 stacks ONCE and cache to MAT; reuse on subsequent runs.
% modeTag identifies your setting (e.g., 'none_default_all_40x32').

    if nargin < 4, cfg = kws_config(); end
    assert(numel(fileList) == numel(labels));

    outRoot = fullfile(cfg.paths.outputDir, 'feature_cache');
    if ~exist(outRoot,'dir'), mkdir(outRoot); end
    cacheName = sprintf('X_%s.mat', modeTag);
    cachePath = fullfile(outRoot, cacheName);

    if exist(cachePath,'file')
        S = load(cachePath, 'X', 'Y', '-mat');
        X = S.X;  Y = S.Y;
        fprintf('Loaded cached features: %s  [%s]\n', cachePath, mat2str(size(X)));
        return;
    end

    % Geometry check
    B = cfg.features.baseBands;    % 40
    W = cfg.features.targetFrames; % 32 for Sainath
    assert(B==40 && W==32, 'Expecting 40x32 geometry for Sainath recreation.');

    % Progress & timing
    N = numel(fileList);
    fprintf('Extracting features (%d files) for %s...\n', N, modeTag);
    t0 = tic;

    % Batch in chunks to control memory
    chunk = 1024;
    X = zeros(B, W, 1, N, 'single');
    Y = categorical(labels);
    for i = 1:chunk:N
        j = min(N, i+chunk-1);
        batch = fileList(i:j);
        Xi = extractFeatures(batch, 'all', 'default', cfg);  % your existing extractor
        % Xi expected size: [B, W, 1, numel(batch)]
        if size(Xi,1)~=B || size(Xi,2)~=W || size(Xi,3)~=1
            error('Extractor returned %s, expected [%d %d 1 *].', mat2str(size(Xi)), B, W);
        end
        X(:,:,:,i:j) = Xi;
        if mod(i, 4096) == 1 || j==N
            fprintf('  %6d / %6d (%.1f%%) in %.1f s\n', j, N, 100*j/N, toc(t0));
        end
    end

    save(cachePath, 'X', 'Y', '-v7.3');
    fprintf('Saved cache: %s  (%.1f s total)\n', cachePath, toc(t0));
end
