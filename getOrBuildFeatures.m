function [X, Y, cachePath] = getOrBuildFeatures(modeTag, fileList, labels, cfg, genderType, melMode)
% getOrBuildFeatures: extract features once and cache to disk.
% Cache key includes dataset + feature geometry + gender/mel settings.

if nargin < 4 || isempty(cfg), cfg = kws_config(); end
if nargin < 5 || isempty(genderType), genderType = 'all'; end
if nargin < 6 || isempty(melMode), melMode = 'default'; end

assert(numel(fileList) == numel(labels));

B = cfg.features.baseBands;
W = cfg.features.targetFrames;

dsTag = lower(string(cfg.dataset.version));
key = sprintf('%s__%s__%s__%s__%dx%d', dsTag, modeTag, lower(genderType), lower(melMode), B, W);

outRoot = fullfile(cfg.paths.cacheDir, 'feature_cache');
if ~exist(outRoot,'dir'), mkdir(outRoot); end
cachePath = fullfile(outRoot, sprintf('X_%s.mat', key));

if cfg.cache.enableFeatureCache && exist(cachePath,'file')
    S = load(cachePath, 'X', 'Y', '-mat');
    X = S.X; Y = S.Y;
    fprintf('Loaded cached features: %s  [%s]\n', cachePath, mat2str(size(X)));
    return;
end

N = numel(fileList);
if ~cfg.runtime.quiet
    fprintf('Extracting features (%d files) for %s...\n', N, key);
end
t0 = tic;

chunk = 1024;
X = zeros(B, W, 1, N, 'single');
Y = categorical(labels);

for i = 1:chunk:N
    j = min(N, i+chunk-1);
    batch = fileList(i:j);

    [Xi, validIdx] = extractFeatures(batch, genderType, melMode, cfg); %#ok<ASGLU>
    if size(Xi,1)~=B || size(Xi,2)~=W || size(Xi,3)~=1
        error('Extractor returned %s, expected [%d %d 1 *].', mat2str(size(Xi)), B, W);
    end
    X(:,:,:,i:j) = Xi;

    if mod(i, 4096) == 1 || j==N
        fprintf('  %6d / %6d (%.1f%%) in %.1f s\n', j, N, 100*j/N, toc(t0));
    end
end

if cfg.cache.enableFeatureCache
    save(cachePath, 'X', 'Y', '-v7.3');
    fprintf('Saved cache: %s  (%.1f s total)\n', cachePath, toc(t0));
end

end
